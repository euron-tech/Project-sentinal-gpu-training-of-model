# One-time bootstrap (run by `python scripts/ops.py up`, local state in this folder):
#   - S3 bucket for the main stack's Terraform state
#   - GitHub Actions OIDC provider (only if the account does not have one yet)
#   - IAM role that GitHub Actions assumes to deploy / destroy (only from this repo's main branch or
#     the "production" environment). It can manage IAM only for names starting with "<project>-".
# Destroy it last: `python scripts/ops.py down --all`.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.95" }
  }
}

variable "region" {
  type    = string
  default = "ap-south-1"
}
variable "project" {
  type    = string
  default = "sentinel"
}
variable "github_repo" {
  description = "owner/repo that may deploy, e.g. euron-tech/Project-sentinal-gpu-training-of-model ('' = no CI role)"
  type        = string
  default     = ""
}
variable "create_oidc_provider" {
  description = "false when the account already has token.actions.githubusercontent.com (ops.py detects this)"
  type        = bool
  default     = true
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = var.project, ManagedBy = "terraform", Stack = "bootstrap" }
  }
}

data "aws_caller_identity" "me" {}

locals {
  account = data.aws_caller_identity.me.account_id
  ci      = var.github_repo != ""
  oidc_arn = var.create_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : (
  "arn:aws:iam::${local.account}:oidc-provider/token.actions.githubusercontent.com")
}

# ---------------------------------------------------------------- state bucket
resource "aws_s3_bucket" "state" {
  bucket        = "${var.project}-tfstate-${local.account}-${var.region}"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------- GitHub OIDC + deploy role
resource "aws_iam_openid_connect_provider" "github" {
  count           = var.create_oidc_provider ? 1 : 0
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github" {
  count = local.ci ? 1 : 0
  name  = "${var.project}-github-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.oidc_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = { "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com" }
        StringLike = { "token.actions.githubusercontent.com:sub" = [
          "repo:${var.github_repo}:ref:refs/heads/main",
          "repo:${var.github_repo}:environment:production",
        ] }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "power" {
  count      = local.ci ? 1 : 0
  role       = aws_iam_role.github[0].name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

resource "aws_iam_role_policy" "iam_scoped" {
  count = local.ci ? 1 : 0
  name  = "${var.project}-iam-scoped"
  role  = aws_iam_role.github[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ManageOnlyProjectIam"
        Effect = "Allow"
        Action = "iam:*"
        Resource = [
          "arn:aws:iam::${local.account}:role/${var.project}-*",
          "arn:aws:iam::${local.account}:policy/${var.project}-*",
          "arn:aws:iam::${local.account}:instance-profile/${var.project}-*",
        ]
      },
      { Sid = "ReadIam", Effect = "Allow", Action = ["iam:Get*", "iam:List*"], Resource = "*" },
      { Sid = "ServiceLinkedRoles", Effect = "Allow", Action = "iam:CreateServiceLinkedRole", Resource = "*" },
    ]
  })
}

output "state_bucket" { value = aws_s3_bucket.state.bucket }
output "github_role_arn" { value = local.ci ? aws_iam_role.github[0].arn : "" }
output "region" { value = var.region }
