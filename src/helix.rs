#![allow(clippy::result_large_err)]

use anyhow::{Context, Result};
use serde_json::{Map, Value};

/// Thin client for the HelixDB v2 HTTP query API.
#[derive(Clone)]
pub struct HelixClient {
    http: reqwest::Client,
    base_url: String,
}

impl HelixClient {
    pub fn new(base_url: &str) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .context("build helix http client")?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// Fail fast when the server is unreachable.
    pub async fn health(&self) -> Result<()> {
        let url = format!("{}/healthz", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("reach helix at {url}"))?;
        if !resp.status().is_success() {
            anyhow::bail!("helix unhealthy: HTTP {}", resp.status());
        }
        Ok(())
    }

    /// POST one single-entry batch and return the parsed JSON body.
    pub async fn query(&self, name: &str, request_type: &str, root: Value) -> Result<Value> {
        let mut batch = Map::new();
        batch.insert(
            "entries".to_string(),
            Value::Array(vec![Value::Object({
                let mut entry = Map::new();
                entry.insert("name".to_string(), Value::String(name.to_string()));
                entry.insert("root".to_string(), root);
                let mut wrapper = Map::new();
                wrapper.insert("query".to_string(), Value::Object(entry));
                wrapper
            })]),
        );
        batch.insert(
            "returns".to_string(),
            Value::Array(vec![Value::String(name.to_string())]),
        );
        let mut query = Map::new();
        query.insert(request_type.to_string(), Value::Object(batch));
        let mut body = Map::new();
        body.insert(
            "request_type".to_string(),
            Value::String(request_type.to_string()),
        );
        body.insert("query_name".to_string(), Value::String(name.to_string()));
        body.insert("query".to_string(), Value::Object(query));

        let url = format!("{}/v2/query", self.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("post {url}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("helix query `{name}` failed: HTTP {status}: {text}");
        }
        serde_json::from_str(&text).with_context(|| format!("parse helix reply: {text}"))
    }
}

/// Property value helpers matching the Helix JSON encoding.
pub fn prop_string(s: &str) -> Value {
    serde_json::json!({"value": {"string": s}})
}

pub fn prop_f32(f: f32) -> Value {
    serde_json::json!({"value": {"f32": f}})
}

pub fn prop_i64(i: i64) -> Value {
    serde_json::json!({"value": {"i64": i}})
}

pub fn prop_f32_array(v: &[f32]) -> Value {
    serde_json::json!({"value": {"f32_array": v}})
}

/// Root step builders. Shapes follow the HelixDB SDK query guides.
pub mod steps {
    use super::prop_f32_array;
    use serde_json::{json, Value};

    pub fn create_vector_index(
        label: &str,
        property: &str,
        dimension: usize,
        metric: &str,
    ) -> Value {
        json!({
            "create_index": {
                "spec": {"node_vector": {
                    "label": label,
                    "property": property,
                    "dimension": dimension,
                    "metric": metric,
                }},
                "if_not_exists": true,
            }
        })
    }

    pub fn add_node(label: &str, properties: Vec<(String, Value)>) -> Value {
        let props: Vec<Value> = properties
            .into_iter()
            .map(|(k, v)| Value::Array(vec![Value::String(k), v]))
            .collect();
        json!({"add_n": {"label": label, "properties": props}})
    }

    /// Scoped vector search: results always belong to `agent_id`, with an
    /// optional extra restriction to one namespace within that agent.
    pub fn vector_search(
        label: &str,
        property: &str,
        vector: &[f32],
        k: u32,
        agent_id: &str,
        namespace: Option<&str>,
    ) -> Value {
        let mut predicates = vec![json!({"eq": {
            "left": {"property": "agent_id"},
            "right": {"constant": {"string": agent_id}},
        }})];
        if let Some(ns) = namespace {
            predicates.push(json!({"eq": {
                "left": {"property": "namespace"},
                "right": {"constant": {"string": ns}},
            }}));
        }
        let predicate = if predicates.len() == 1 {
            predicates.pop().unwrap()
        } else {
            json!({"and": {"predicates": predicates}})
        };
        json!({
            "vector_search_nodes_within": {
                "input": {"nodes_where": {"predicate": predicate}},
                "label": label,
                "property": property,
                "query_vector": prop_f32_array(vector),
                "k": {"literal": k},
            }
        })
    }

    pub fn value_map(input: Value, properties: &[&str]) -> Value {
        json!({"value_map": {"input": input, "properties": properties}})
    }

    pub fn drop_by_id(id: u64) -> Value {
        json!({"drop": {"input": {"nodes": {"reference": {"ids": [id]}}}}})
    }

    pub fn node_props(id: u64, properties: &[&str]) -> Value {
        json!({"value_map": {
            "input": {"nodes": {"reference": {"ids": [id]}}},
            "properties": properties,
        }})
    }

    pub fn count(label: &str, agent_id: &str, namespace: Option<&str>) -> Value {
        let mut predicates = vec![
            json!({"eq": {
                "left": {"property": "$label"},
                "right": {"constant": {"string": label}},
            }}),
            json!({"eq": {
                "left": {"property": "agent_id"},
                "right": {"constant": {"string": agent_id}},
            }}),
        ];
        if let Some(ns) = namespace {
            predicates.push(json!({"eq": {
                "left": {"property": "namespace"},
                "right": {"constant": {"string": ns}},
            }}));
        }
        let predicate = json!({"and": {"predicates": predicates}});
        json!({"count": {"input": {"nodes_where": {"predicate": predicate}}}})
    }
}

#[cfg(test)]
mod tests {
    use super::steps;
    use serde_json::json;

    #[test]
    fn vector_search_always_scopes_to_agent() {
        let root = steps::vector_search("Memory", "embedding", &[0.1, 0.2], 5, "agent-a", None);
        let scoped = &root["vector_search_nodes_within"];
        assert!(scoped.is_object(), "must use scoped search: {root}");
        assert_eq!(
            scoped["input"]["nodes_where"]["predicate"],
            json!({"eq": {
                "left": {"property": "agent_id"},
                "right": {"constant": {"string": "agent-a"}},
            }}),
        );
    }

    #[test]
    fn vector_search_ands_namespace_within_agent() {
        let root = steps::vector_search(
            "Memory",
            "embedding",
            &[0.1, 0.2],
            5,
            "agent-a",
            Some("proj"),
        );
        let predicate = &root["vector_search_nodes_within"]["input"]["nodes_where"]["predicate"];
        let predicates = predicate["and"]["predicates"]
            .as_array()
            .expect("namespace filter must AND with the agent filter");
        assert_eq!(predicates.len(), 2);
        let const_str =
            |p: &serde_json::Value| p["eq"]["right"]["constant"]["string"].clone();
        assert!(predicates.iter().any(|p| const_str(p) == json!("agent-a")));
        assert!(predicates.iter().any(|p| const_str(p) == json!("proj")));
    }

    #[test]
    fn count_scopes_to_agent() {
        let root = steps::count("Memory", "agent-a", Some("proj"));
        let predicates = root["count"]["input"]["nodes_where"]["predicate"]["and"]["predicates"]
            .as_array()
            .expect("count must filter by label, agent and namespace");
        let flat: Vec<String> = predicates.iter().map(|p| p.to_string()).collect();
        assert!(flat.iter().any(|p| p.contains("agent-a")), "{flat:?}");
        assert!(flat.iter().any(|p| p.contains("\"proj\"")), "{flat:?}");
    }

    #[test]
    fn node_props_targets_single_id() {
        let root = steps::node_props(42, &["agent_id"]);
        assert_eq!(
            root["value_map"]["input"]["nodes"]["reference"]["ids"],
            json!([42]),
        );
        assert_eq!(root["value_map"]["properties"], json!(["agent_id"]));
    }
}
