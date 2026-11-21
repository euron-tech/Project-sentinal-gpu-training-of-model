# ECR + S3 dataset bucket + secrets + IAM + ONE ECS Fargate service (no EC2 instances to manage).
# The task runs two containers that talk over localhost:
#   ui      :8501  Streamlit dashboard (Live Flow, Analytics, Benchmark, tests, Decision)
#   stream  :8003  Confluent Kafka producer (auto mode, button-driven) + scorer + analytics API + flow page
# The models are NOT on AWS: the scorer and the dashboard call LoRA / XGBoost / QLoRA on the Hyperstack VM.

data "aws_caller_identity" "me" {}

# ---------------------------------------------------------------- registries + dataset bucket
resource "aws_ecr_repository" "this" {
  for_each             = toset(["ui", "stream"])
  name                 = "${local.name}/${each.key}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # `terraform destroy` removes the images too
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "this" {
  for_each   = aws_ecr_repository.this
  repository = each.value.name
  policy = jsonencode({ rules = [{
    rulePriority = 1, description = "keep the last 10 images"
    selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
    action       = { type = "expire" }
  }] })
}

# data/sft/holdout/*.jsonl.gz: labelled dataset the producer replays (copied once by scripts/ops.py)
resource "aws_s3_bucket" "data" {
  bucket        = "${local.name}-data-${data.aws_caller_identity.me.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

# ---------------------------------------------------------------- secrets (SSM SecureString, injected by ECS)
resource "random_password" "control_key" {
  length  = 40
  special = false
}

locals {
  secrets = {
    model-api-key    = var.model_api_key == "" ? "unset" : var.model_api_key
    control-key      = random_password.control_key.result
    kafka-api-key    = confluent_api_key.app.id
    kafka-api-secret = confluent_api_key.app.secret
  }
}

resource "aws_ssm_parameter" "secret" {
  for_each = local.secrets
  name     = "/${local.name}/${each.key}"
  type     = "SecureString"
  value    = each.value
}

# ---------------------------------------------------------------- IAM
data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_secrets" {
  name = "${local.name}-secrets"
  role = aws_iam_role.execution.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Action = ["ssm:GetParameters"], Resource = [for p in aws_ssm_parameter.secret : p.arn]
  }] })
}

resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy" "task" {
  name = "${local.name}-task"
  role = aws_iam_role.task.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = aws_s3_bucket.data.arn },
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.data.arn}/*" },
    { Effect = "Allow", Action = ["sns:Publish"], Resource = aws_sns_topic.actions.arn },
    { Effect = "Allow", Action = ["cloudwatch:PutMetricData"], Resource = "*",
    Condition = { StringEquals = { "cloudwatch:namespace" = local.cw_namespace } } },
    { Effect = "Allow", Action = ["cloudwatch:GetMetricData", "cloudwatch:ListMetrics"], Resource = "*" },
    { Effect = "Allow", Resource = "*", Action = [ # ECS Exec (aws ecs execute-command)
    "ssmmessages:CreateControlChannel", "ssmmessages:CreateDataChannel", "ssmmessages:OpenControlChannel", "ssmmessages:OpenDataChannel"] },
  ] })
}

# ---------------------------------------------------------------- cluster
resource "aws_ecs_cluster" "this" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled" # CPU / memory per task and container for the dashboard
  }
  depends_on = [aws_cloudwatch_log_group.container_insights]
}

# Container Insights would create this log group on its own (and leave it behind after destroy).
resource "aws_cloudwatch_log_group" "container_insights" {
  name              = "/aws/ecs/containerinsights/${local.name}/performance"
  retention_in_days = 1
}

# ---------------------------------------------------------------- task + service
locals {
  cw_namespace = "Sentinel/${var.environment}"
  service_name = "${local.name}-app"
  ssm          = { for k, p in aws_ssm_parameter.secret : k => p.arn }
  model_url    = { for m, port in { lora = 8000, xgb = 8001, qlora = 8002 } : m => var.hyperstack_host == "" ? "" : "http://${var.hyperstack_host}:${port}" }

  containers = {
    stream = {
      port = 8003, cpu = 512, memory = 768
      env = {
        KAFKA_BOOTSTRAP       = local.kafka_bootstrap, XGB_URL = local.model_url.xgb, LORA_URL = local.model_url.lora
        QLORA_URL             = local.model_url.qlora, CHAMPION = var.champion, CW_NAMESPACE = local.cw_namespace
        SNS_ACTIONS_TOPIC_ARN = aws_sns_topic.actions.arn, STORE_PATH = "/data/sentinel.db", STORE_DIR = "/data"
        AUTO_START            = "0", EVENTS_PER_SEC = tostring(var.events_per_sec), FRAUD_RATE = tostring(var.fraud_rate)
        PRODUCER_SOURCE       = var.producer_source, DATASET_URI = "s3://${aws_s3_bucket.data.bucket}/data/sft/holdout"
      }
      secrets = {
        SENTINEL_API_KEY = local.ssm["model-api-key"], CONTROL_API_KEY = local.ssm["control-key"]
        KAFKA_API_KEY    = local.ssm["kafka-api-key"], KAFKA_API_SECRET = local.ssm["kafka-api-secret"]
      }
    }
    ui = {
      port = 8501, cpu = 512, memory = 1024
      env = {
        LORA_URL    = local.model_url.lora, QLORA_URL = local.model_url.qlora, XGB_URL = local.model_url.xgb
        STREAM_URL  = "http://localhost:8003", PUBLIC_STREAM_URL = "/live", UI_LOCK_CONNECTIONS = "1"
        ECS_CLUSTER = local.name, ECS_SERVICES = local.service_name, CW_NAMESPACE = local.cw_namespace
      }
      secrets = {
        LLM_API_KEY     = local.ssm["model-api-key"], XGB_API_KEY = local.ssm["model-api-key"]
        CONTROL_API_KEY = local.ssm["control-key"]
      }
    }
  }
}

resource "aws_ecs_task_definition" "app" {
  family                   = local.service_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([for name, c in local.containers : {
    name              = name
    image             = "${aws_ecr_repository.this[name].repository_url}:${var.image_tag}"
    essential         = true
    cpu               = c.cpu
    memoryReservation = c.memory
    portMappings      = [{ containerPort = c.port, protocol = "tcp" }]
    environment       = [for k, v in merge({ AWS_REGION = var.region }, c.env) : { name = k, value = v }]
    secrets           = [for k, v in c.secrets : { name = k, valueFrom = v }]
    linuxParameters   = { initProcessEnabled = true }
    stopTimeout       = 30
    healthCheck = {
      command     = ["CMD-SHELL", name == "ui" ? "curl -fs http://localhost:8501/_stcore/health || exit 1" : "curl -fs http://localhost:8003/live/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app[name].name
        awslogs-region        = var.region
        awslogs-stream-prefix = name
      }
    }
  }])
}

resource "aws_ecs_service" "app" {
  name            = local.service_name
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1 # one task: the stream scorer owns the analytics store (rebuilt from Kafka on restart)
  launch_type     = "FARGATE"
  # stop the old task before starting the new one so two scorers never write analytics at the same time
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100
  enable_execute_command             = true
  health_check_grace_period_seconds  = 120
  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true # pulls from ECR + reaches Confluent / Hyperstack without a NAT gateway
  }
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "ui"
    container_port   = 8501
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.stream.arn
    container_name   = "stream"
    container_port   = 8003
  }
  depends_on = [aws_lb_listener.http, aws_lb_listener_rule.live, aws_iam_role_policy.task, confluent_kafka_topic.this]
}
