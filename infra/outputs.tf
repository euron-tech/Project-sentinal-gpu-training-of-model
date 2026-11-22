locals {
  base_url = "${local.https ? "https" : "http"}://${aws_lb.this.dns_name}"
}

output "dashboard_url" { value = local.base_url }
output "flow_url" { value = "${local.base_url}/live/flow" }
output "analytics_api" { value = "${local.base_url}/live/api/summary?minutes=60" }
output "cloudwatch_dashboard" {
  value = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards/dashboard/${local.name}"
}
output "data_bucket" { value = aws_s3_bucket.data.bucket }
output "ecr" { value = { for k, r in aws_ecr_repository.this : k => r.repository_url } }
output "cluster" { value = aws_ecs_cluster.this.name }
output "services" { value = [aws_ecs_service.app.name] }
output "kinesis_streams" { value = { for k, s in aws_kinesis_stream.this : k => s.name } }
output "model_urls" { value = local.model_url }
output "actions_topic_arn" { value = aws_sns_topic.actions.arn }
output "alerts_topic_arn" { value = aws_sns_topic.alerts.arn }
