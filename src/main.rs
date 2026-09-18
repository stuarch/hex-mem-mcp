//! hex-mem-mcp: agent long-term memory MCP server backed by HelixDB.
//!
//! Configuration (environment variables):
//! - `HEX_MEM_HELIX_URL`: HelixDB server base URL (default http://127.0.0.1:6969)
//! - `HEX_MEM_EMBEDDING_BASE_URL`: OpenAI-compatible API base URL
//!   (default http://127.0.0.1:51480/v1)
//! - `HEX_MEM_EMBEDDING_MODEL`: embedding model name (required)
//! - `HEX_MEM_EMBEDDING_API_KEY`: API key, if the endpoint needs one (default "dummy")

mod embed;
mod helix;
mod memory;

use anyhow::{Context, Result};
use rmcp::{transport::stdio, ServiceExt};
use tracing_subscriber::EnvFilter;

use crate::embed::Embedder;
use crate::helix::{steps, HelixClient};
use crate::memory::Memory;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let helix_url = env_or("HEX_MEM_HELIX_URL", "http://127.0.0.1:6969");
    let embed_base = env_or("HEX_MEM_EMBEDDING_BASE_URL", "http://127.0.0.1:51480/v1");
    let embed_model = std::env::var("HEX_MEM_EMBEDDING_MODEL")
        .context("HEX_MEM_EMBEDDING_MODEL must be set to the embedding model name")?;
    let embed_key = env_or("HEX_MEM_EMBEDDING_API_KEY", "dummy");

    let helix = HelixClient::new(&helix_url)?;
    helix.health().await.with_context(|| {
        format!("cannot reach HelixDB at {helix_url}; is helix-server running?")
    })?;
    tracing::info!("helix reachable at {helix_url}");

    let embedder = Embedder::new(&embed_base, &embed_model, &embed_key)?;
    let probe_vec = embedder
        .embed(&["readiness probe".to_string()])
        .await
        .with_context(|| {
            format!("cannot get embeddings from {embed_base} with model {embed_model}")
        })?
        .into_iter()
        .next()
        .context("embedding model returned no vector")?;
    let dim = probe_vec.len();
    if dim == 0 {
        anyhow::bail!("embedding model returned an empty vector");
    }
    tracing::info!("embedding model {embed_model} dimension {dim}");

    // Bootstrap the vector index once; a no-op when it already exists.
    // NOTE: the embedding model (and therefore the dimension) must stay
    // fixed afterwards, or writes will fail dimension checks.
    let root = steps::create_vector_index("Memory", "embedding", dim, "cosine");
    helix
        .query("index", "write", root)
        .await
        .context("create Memory embedding index")?;
    // Index creation is async ("a new generation remains hidden until
    // validation and activation succeed"), so poll a probe search until the
    // planner sees it.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        let root = steps::vector_search("Memory", "embedding", &probe_vec, 1, None);
        match helix.query("probe", "read", root).await {
            Ok(_) => break,
            Err(e) if e.to_string().contains("index_not_found") => {
                if std::time::Instant::now() > deadline {
                    anyhow::bail!("timed out waiting for Memory index activation");
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
            Err(e) => return Err(e).context("probe Memory index"),
        }
    }
    tracing::info!("memory index ready");

    let service = Memory::new(helix, embedder)
        .serve(stdio())
        .await
        .inspect_err(|e| tracing::error!("serving error: {e:?}"))?;
    service.waiting().await?;
    Ok(())
}
