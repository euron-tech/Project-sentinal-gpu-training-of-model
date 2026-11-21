# Confluent Cloud Kafka, created and destroyed with this stack: its own environment, a Basic cluster in the
# same AWS region, the three topics, and a service account + API key used only by the stream service.

resource "confluent_environment" "this" {
  display_name = local.name
  stream_governance {
    package = "ESSENTIALS"
  }
}

resource "confluent_kafka_cluster" "this" {
  display_name = local.name
  availability = "SINGLE_ZONE"
  cloud        = "AWS"
  region       = var.region
  basic {}
  environment {
    id = confluent_environment.this.id
  }
}

resource "confluent_service_account" "app" {
  display_name = "${local.name}-app"
  description  = "SENTINEL stream service (producer + scorer)"
}

resource "confluent_role_binding" "app" {
  principal   = "User:${confluent_service_account.app.id}"
  role_name   = "CloudClusterAdmin"
  crn_pattern = confluent_kafka_cluster.this.rbac_crn
}

resource "confluent_api_key" "app" {
  display_name = "${local.name}-app"
  owner {
    id          = confluent_service_account.app.id
    api_version = confluent_service_account.app.api_version
    kind        = confluent_service_account.app.kind
  }
  managed_resource {
    id          = confluent_kafka_cluster.this.id
    api_version = confluent_kafka_cluster.this.api_version
    kind        = confluent_kafka_cluster.this.kind
    environment {
      id = confluent_environment.this.id
    }
  }
  depends_on = [confluent_role_binding.app]
}

locals {
  kafka_bootstrap = replace(confluent_kafka_cluster.this.bootstrap_endpoint, "SASL_SSL://", "")
  topics = { # name => retention (ms). decisions are kept 7 days: the analytics store is rebuilt from them
    "sentinel.transactions" = 86400000
    "sentinel.decisions"    = 604800000
    "sentinel.actions"      = 604800000
  }
}

resource "confluent_kafka_topic" "this" {
  for_each         = local.topics
  topic_name       = each.key
  partitions_count = 3
  config           = { "retention.ms" = tostring(each.value) }
  rest_endpoint    = confluent_kafka_cluster.this.rest_endpoint
  kafka_cluster {
    id = confluent_kafka_cluster.this.id
  }
  credentials {
    key    = confluent_api_key.app.id
    secret = confluent_api_key.app.secret
  }
}
