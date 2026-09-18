//! MCP tools: memory_store / memory_recall / memory_forget / memory_stats.

use std::time::{SystemTime, UNIX_EPOCH};

use rmcp::{
    handler::server::wrapper::Parameters, model::*, schemars, tool, tool_handler, tool_router,
    ErrorData as McpError, ServerHandler,
};

use crate::embed::Embedder;
use crate::helix::{prop_f32, prop_f32_array, prop_i64, prop_string, steps, HelixClient};

const LABEL: &str = "Memory";
const EMBEDDING_PROP: &str = "embedding";
const DEFAULT_NAMESPACE: &str = "default";

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn text_result(text: String) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
}

fn internal<E: std::fmt::Display>(err: E) -> McpError {
    McpError::internal_error(err.to_string(), None)
}

/// Agent identity is cooperative (any local caller can claim any id) but
/// structurally enforced: every query filters on it, so one agent can never
/// see another agent's memories through these tools.
fn require_agent_id(id: &str) -> Result<String, McpError> {
    let id = id.trim();
    if id.is_empty() {
        return Err(McpError::invalid_params("agent_id must not be empty", None));
    }
    Ok(id.to_string())
}

/// Read a string property whether Helix returns it decoded or still wrapped
/// in its `{"value": {"string": ...}}` encoding.
fn prop_as_str(v: &serde_json::Value) -> Option<&str> {
    v.as_str()
        .or_else(|| v.get("value")?.get("string")?.as_str())
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StoreArgs {
    /// Identity of the calling agent. The memory is private to this id.
    pub agent_id: String,
    /// The memory text to store. Write it as a self-contained fact or note.
    pub text: String,
    /// Namespace isolating memories per project within this agent. Defaults to "default".
    pub namespace: Option<String>,
    /// Importance from 0.0 to 1.0. Defaults to 0.5.
    pub importance: Option<f32>,
    /// Short free-form tags for later filtering.
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallArgs {
    /// Identity of the calling agent. Only this agent's memories are searched.
    pub agent_id: String,
    /// Natural-language query describing what to recall.
    pub query: String,
    /// Restrict recall to one namespace. Omit to search all namespaces.
    pub namespace: Option<String>,
    /// Max memories to return (1-50). Defaults to 5.
    pub top_k: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ForgetArgs {
    /// Identity of the calling agent. Must own the memory to delete it.
    pub agent_id: String,
    /// The numeric id of the memory to delete (see memory_recall results).
    pub id: u64,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StatsArgs {
    /// Identity of the calling agent. Only this agent's memories are counted.
    pub agent_id: String,
    /// Count memories in one namespace only. Omit to count all namespaces.
    pub namespace: Option<String>,
}

#[derive(Clone)]
pub struct Memory {
    helix: HelixClient,
    embedder: Embedder,
}

#[tool_router]
impl Memory {
    pub fn new(helix: HelixClient, embedder: Embedder) -> Self {
        Self { helix, embedder }
    }

    #[tool(description = "Store one memory (a self-contained fact or note) for later recall.")]
    async fn memory_store(
        &self,
        Parameters(args): Parameters<StoreArgs>,
    ) -> Result<CallToolResult, McpError> {
        if args.text.trim().is_empty() {
            return Err(McpError::invalid_params("text must not be empty", None));
        }
        let agent = require_agent_id(&args.agent_id)?;
        let namespace = args
            .namespace
            .unwrap_or_else(|| DEFAULT_NAMESPACE.to_string());
        let importance = args.importance.unwrap_or(0.5).clamp(0.0, 1.0);
        let tags = args.tags.unwrap_or_default().join(",");
        let vecs = self
            .embedder
            .embed(std::slice::from_ref(&args.text))
            .await
            .map_err(internal)?;
        let embedding = vecs
            .into_iter()
            .next()
            .ok_or_else(|| McpError::internal_error("embedding model returned no vector", None))?;
        let root = steps::add_node(
            LABEL,
            vec![
                ("text".to_string(), prop_string(&args.text)),
                ("agent_id".to_string(), prop_string(&agent)),
                ("namespace".to_string(), prop_string(&namespace)),
                ("importance".to_string(), prop_f32(importance)),
                ("created_at".to_string(), prop_i64(now_epoch())),
                ("tags".to_string(), prop_string(&tags)),
                ("embedding".to_string(), prop_f32_array(&embedding)),
            ],
        );
        let reply = self
            .helix
            .query("stored", "write", root)
            .await
            .map_err(internal)?;
        let id = reply
            .get("stored")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|n| n.get("$id"))
            .and_then(|v| v.as_u64());
        match id {
            Some(id) => text_result(
                serde_json::json!({
                    "id": id,
                    "agent_id": agent,
                    "namespace": namespace,
                    "chars": args.text.chars().count(),
                })
                .to_string(),
            ),
            None => Err(internal(format!("unexpected helix reply: {reply}"))),
        }
    }

    #[tool(
        description = "Recall memories semantically similar to the query. Results arrive in relevance order."
    )]
    async fn memory_recall(
        &self,
        Parameters(args): Parameters<RecallArgs>,
    ) -> Result<CallToolResult, McpError> {
        if args.query.trim().is_empty() {
            return Err(McpError::invalid_params("query must not be empty", None));
        }
        let agent = require_agent_id(&args.agent_id)?;
        let top_k = args.top_k.unwrap_or(5).clamp(1, 50);
        let vecs = self
            .embedder
            .embed(std::slice::from_ref(&args.query))
            .await
            .map_err(internal)?;
        let embedding = vecs
            .into_iter()
            .next()
            .ok_or_else(|| McpError::internal_error("embedding model returned no vector", None))?;
        let search = steps::vector_search(
            LABEL,
            EMBEDDING_PROP,
            &embedding,
            top_k,
            &agent,
            args.namespace.as_deref(),
        );
        let root = steps::value_map(
            search,
            &["$id", "agent_id", "text", "namespace", "importance", "created_at"],
        );
        let reply = self
            .helix
            .query("hits", "read", root)
            .await
            .map_err(internal)?;
        let hits = reply
            .get("hits")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let results: Vec<serde_json::Value> = hits
            .iter()
            .map(|h| {
                serde_json::json!({
                    "id": h.get("$id"),
                    "agent_id": h.get("agent_id"),
                    "text": h.get("text"),
                    "namespace": h.get("namespace"),
                    "importance": h.get("importance"),
                    "created_at": h.get("created_at"),
                })
            })
            .collect();
        text_result(serde_json::json!({"count": results.len(), "results": results}).to_string())
    }

    #[tool(description = "Delete one memory by its numeric id. Only the owning agent may delete it.")]
    async fn memory_forget(
        &self,
        Parameters(args): Parameters<ForgetArgs>,
    ) -> Result<CallToolResult, McpError> {
        let agent = require_agent_id(&args.agent_id)?;
        let root = steps::node_props(args.id, &["agent_id"]);
        let reply = self
            .helix
            .query("owner", "read", root)
            .await
            .map_err(internal)?;
        let owner = reply
            .get("owner")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|n| n.get("agent_id"))
            .and_then(prop_as_str);
        // Deliberately ambiguous: a wrong id and another agent's id look the same.
        if owner != Some(agent.as_str()) {
            return Err(McpError::invalid_params(
                format!("memory {} not found or belongs to another agent", args.id),
                None,
            ));
        }
        let root = steps::drop_by_id(args.id);
        self.helix
            .query("gone", "write", root)
            .await
            .map_err(internal)?;
        text_result(serde_json::json!({"forgot": args.id}).to_string())
    }

    #[tool(description = "Count this agent's stored memories, optionally restricted to one namespace.")]
    async fn memory_stats(
        &self,
        Parameters(args): Parameters<StatsArgs>,
    ) -> Result<CallToolResult, McpError> {
        let agent = require_agent_id(&args.agent_id)?;
        let root = steps::count(LABEL, &agent, args.namespace.as_deref());
        let reply = self
            .helix
            .query("total", "read", root)
            .await
            .map_err(internal)?;
        let count = reply.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
        text_result(
            serde_json::json!({"agent_id": agent, "namespace": args.namespace, "count": count})
                .to_string(),
        )
    }
}

#[tool_handler]
impl ServerHandler for Memory {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_instructions(
                "Agent long-term memory backed by HelixDB. Every tool requires agent_id: \
             memories are private to that id, and each tool only ever sees the calling \
             agent's own memories. Tools: memory_store saves one self-contained fact \
             (optionally namespaced per project); memory_recall finds similar memories; \
             memory_forget deletes by id; memory_stats counts stored memories."
                    .to_string(),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::{prop_as_str, require_agent_id};
    use serde_json::json;

    #[test]
    fn agent_id_must_be_nonempty() {
        assert!(require_agent_id("").is_err());
        assert!(require_agent_id("   ").is_err());
        assert_eq!(require_agent_id("  agent-a ").unwrap(), "agent-a");
    }

    #[test]
    fn prop_as_str_reads_both_encodings() {
        assert_eq!(prop_as_str(&json!("agent-a")), Some("agent-a"));
        assert_eq!(
            prop_as_str(&json!({"value": {"string": "agent-a"}})),
            Some("agent-a")
        );
        assert_eq!(prop_as_str(&json!(null)), None);
    }
}
