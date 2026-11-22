terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.95" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
  # bucket / key / region are passed by scripts/ops.py (state bucket comes from infra/bootstrap)
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags { tags = local.tags } # every resource is tagged Project=<project>, so it is easy to audit
}
