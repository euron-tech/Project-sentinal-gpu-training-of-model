variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  description = "Prefix of every resource name. The CI role may only manage IAM named <project>-*"
  type        = string
  default     = "sentinel"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "image_tag" {
  description = "Tag of the sentinel images in ECR (CI passes the git SHA)"
  type        = string
  default     = "latest"
}

# ---------------------------------------------------------------- models (all run on the Hyperstack GPU VM)
variable "hyperstack_host" {
  description = "Public IP / DNS of the Hyperstack VM running hyperstack/serve_all.sh (LoRA :8000, XGBoost :8001, QLoRA :8002)"
  type        = string
  default     = ""
}

variable "model_api_key" {
  description = "SENTINEL_API_KEY printed by serve_all.sh on the VM (stored in SSM, never in the image)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "champion" {
  description = "Model whose decision is used on the stream: xgboost | lora | qlora (others run in shadow)"
  type        = string
  default     = "xgboost"
}

# ---------------------------------------------------------------- traffic
variable "events_per_sec" {
  description = "Default auto-traffic rate (changeable live on the dashboard)"
  type        = number
  default     = 2
}

variable "fraud_rate" {
  type    = number
  default = 0.04
}

variable "producer_source" {
  description = "synthetic | dataset | mix (mix = labelled SFT dataset from S3 when present + synthetic fraud rings)"
  type        = string
  default     = "mix"
}

# ---------------------------------------------------------------- Fargate task (dashboard + stream scorer)
variable "task_cpu" {
  type    = number
  default = 1024
}

variable "task_memory" {
  type    = number
  default = 2048
}

# ---------------------------------------------------------------- access + ops
variable "allowed_cidrs" {
  description = "Who can open the dashboard"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "certificate_arn" {
  description = "Optional ACM certificate for HTTPS on the load balancer ('' = HTTP only)"
  type        = string
  default     = ""
}

variable "alert_email" {
  description = "Optional e-mail for CloudWatch alarms (confirm the SNS subscription mail)"
  type        = string
  default     = ""
}

variable "log_retention_days" {
  type    = number
  default = 14
}
