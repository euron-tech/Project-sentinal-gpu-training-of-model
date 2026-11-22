# Logging (one CloudWatch log group per service), alerting (SNS) and a CloudWatch dashboard.

resource "aws_cloudwatch_log_group" "app" {
  for_each          = toset(["ui", "stream"])
  name              = "/ecs/${local.name}/${each.key}"
  retention_in_days = var.log_retention_days
}

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# The "further action" hook: every BLOCK decision is published here (subscribe a Lambda, webhook, queue, ...).
resource "aws_sns_topic" "actions" {
  name = "${local.name}-fraud-actions"
}

locals {
  alarm_actions = [aws_sns_topic.alerts.arn]
  stream_dims   = { Service = "stream" }
}

resource "aws_cloudwatch_metric_alarm" "task_cpu" {
  alarm_name          = "${local.name}-task-cpu-high"
  namespace           = "AWS/ECS"
  metric_name         = "CPUUtilization"
  dimensions          = { ClusterName = aws_ecs_cluster.this.name, ServiceName = aws_ecs_service.app.name }
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 85
  comparison_operator = "GreaterThanThreshold"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "task_memory" {
  alarm_name          = "${local.name}-task-memory-high"
  namespace           = "AWS/ECS"
  metric_name         = "MemoryUtilization"
  dimensions          = { ClusterName = aws_ecs_cluster.this.name, ServiceName = aws_ecs_service.app.name }
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 85
  comparison_operator = "GreaterThanThreshold"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "dashboard_unhealthy" {
  alarm_name          = "${local.name}-dashboard-unhealthy"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  dimensions          = { LoadBalancer = aws_lb.this.arn_suffix, TargetGroup = aws_lb_target_group.ui.arn_suffix }
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 5
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${local.name}-alb-5xx"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  dimensions          = { LoadBalancer = aws_lb.this.arn_suffix }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 20
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "models_unreachable" {
  alarm_name          = "${local.name}-models-unreachable"
  alarm_description   = "Transactions are being held for REVIEW because no Hyperstack model answered (fail closed)"
  namespace           = local.cw_namespace
  metric_name         = "NoModel"
  dimensions          = local.stream_dims
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "stream_lag" {
  alarm_name          = "${local.name}-kinesis-iterator-age"
  alarm_description   = "The scorer is more than 60 s behind the transactions stream"
  namespace           = "AWS/Kinesis"
  metric_name         = "GetRecords.IteratorAgeMilliseconds"
  dimensions          = { StreamName = aws_kinesis_stream.this["transactions"].name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 60000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "fraud_spike" {
  alarm_name          = "${local.name}-fraud-spike"
  alarm_description   = "Unusually many blocked transactions (attack wave)"
  namespace           = local.cw_namespace
  metric_name         = "Blocked"
  dimensions          = local.stream_dims
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 60
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_dashboard" "this" {
  dashboard_name = local.name
  dashboard_body = jsonencode({ widgets = [
    { type = "text", x = 0, y = 0, width = 24, height = 2, properties = {
    markdown = "# SENTINEL ${var.environment}\nAmazon Kinesis → stream scorer (Fargate) → XGBoost (champion) + Qwen LoRA / QLoRA (shadow) on Hyperstack → decision → action · [Open dashboard](http://${aws_lb.this.dns_name}) · [Live flow](http://${aws_lb.this.dns_name}/live/flow)" } },
    { type = "metric", x = 0, y = 2, width = 8, height = 6, properties = {
      title = "Transactions and decisions / min", region = var.region, stat = "Sum", period = 60, view = "timeSeries",
    metrics = [for m in ["Transactions", "Allowed", "Reviewed", "Blocked"] : [local.cw_namespace, m, "Service", "stream"]] } },
    { type = "metric", x = 8, y = 2, width = 8, height = 6, properties = {
      title = "Model latency per transaction (ms)", region = var.region, stat = "Average", period = 60, view = "timeSeries",
    metrics = [for m in ["xgboost", "lora", "qlora"] : [local.cw_namespace, "ScoreLatency", "Model", m]] } },
    { type = "metric", x = 16, y = 2, width = 8, height = 6, properties = {
      title = "Kinesis records in / scorer lag (ms) / missed fraud", region = var.region, period = 60, view = "timeSeries",
      metrics = [["AWS/Kinesis", "IncomingRecords", "StreamName", aws_kinesis_stream.this["transactions"].name, { stat = "Sum" }],
        ["AWS/Kinesis", "GetRecords.IteratorAgeMilliseconds", "StreamName", aws_kinesis_stream.this["transactions"].name, { stat = "Maximum" }],
    [local.cw_namespace, "FraudMissed", "Service", "stream", { stat = "Sum" }]] } },
    { type = "metric", x = 0, y = 8, width = 8, height = 6, properties = {
      title = "Fargate task CPU / memory %", region = var.region, view = "timeSeries", period = 60, stat = "Average",
      metrics = [["AWS/ECS", "CPUUtilization", "ClusterName", local.name, "ServiceName", local.service_name],
    ["AWS/ECS", "MemoryUtilization", "ClusterName", local.name, "ServiceName", local.service_name]] } },
    { type = "metric", x = 8, y = 8, width = 8, height = 6, properties = {
      title = "Container CPU / memory (Container Insights)", region = var.region, view = "timeSeries", period = 60,
    metrics = [[{ expression = "SEARCH('{ECS/ContainerInsights,ClusterName,TaskDefinitionFamily} ClusterName=\"${local.name}\" (MetricName=\"CpuUtilized\" OR MetricName=\"MemoryUtilized\")', 'Average', 60)", id = "ci" }]] } },
    { type = "metric", x = 16, y = 8, width = 8, height = 6, properties = {
      title = "ALB requests / target response time", region = var.region, view = "timeSeries", period = 60,
      metrics = [["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.this.arn_suffix, { stat = "Sum" }],
    ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.this.arn_suffix, { stat = "Average", yAxis = "right" }]] } },
    { type = "log", x = 0, y = 14, width = 24, height = 6, properties = {
      title = "Warnings and errors (all services)", region = var.region, view = "table",
    query = "${join(" | ", [for k in ["ui", "stream"] : "SOURCE '/ecs/${local.name}/${k}'"])} | fields @timestamp, @logStream, @message | filter @message like /(?i)(error|warn|exception|traceback)/ | sort @timestamp desc | limit 50" } },
  ] })
  depends_on = [aws_cloudwatch_log_group.app]
}
