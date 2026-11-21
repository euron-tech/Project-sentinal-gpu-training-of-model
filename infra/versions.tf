terraform {
  required_version = ">= 1.6"
  required_providers {
    aws       = { source = "hashicorp/aws", version = "~> 5.95" }
    random    = { source = "hashicorp/random", version = "~> 3.6" }
    confluent = { source = "confluentinc/confluent", version = "~> 2.0" }
  }
  # bucket / key / region are passed by scripts/ops.py (state bucket comes from infra/bootstrap)
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags { tags = local.tags } # every resource is tagged Project=<project>, so it is easy to audit
}

# Confluent Cloud: credentials come from the env vars CONFLUENT_CLOUD_API_KEY / CONFLUENT_CLOUD_API_SECRET
# (a Cloud resource-management API key, created at confluent.cloud -> API keys).
provider "confluent" {}
