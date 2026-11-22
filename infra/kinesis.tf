# Amazon Kinesis Data Streams: the event bus. One shard each (1 MB/s or 1,000 records/s in, plenty for the demo).
#   <name>-transactions  producer -> scorer
#   <name>-decisions     every scored transaction + its action (APPROVE / STEP_UP_AUTH / BLOCK_AND_ALERT): the
#                        "further action" hook for downstream systems; the analytics store is rebuilt from it
# Both kept 24 h (the default, no extended-retention charge).
# Encrypted with the AWS managed key; destroyed with the stack.

locals {
  streams = { transactions = 24, decisions = 24 } # name => retention hours
}

resource "aws_kinesis_stream" "this" {
  for_each         = local.streams
  name             = "${local.name}-${each.key}"
  retention_period = each.value
  shard_count      = 1
  encryption_type  = "KMS"
  kms_key_id       = "alias/aws/kinesis"
  stream_mode_details {
    stream_mode = "PROVISIONED"
  }
}
