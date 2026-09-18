//! OpenAI-compatible embeddings client (works with llama-server /v1/embeddings).

use anyhow::{Context, Result};
use serde_json::json;

#[derive(Clone)]
pub struct Embedder {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

impl Embedder {
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .context("build embedding http client")?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
        })
    }

    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&json!({"model": self.model, "input": inputs}))
            .send()
            .await
            .with_context(|| format!("post {url}"))?;
        let status = resp.status();
        let body: serde_json::Value = resp.json().await.context("parse embedding reply")?;
        if !status.is_success() {
            anyhow::bail!("embedding request failed: HTTP {status}: {body}");
        }
        let data = body
            .get("data")
            .and_then(|d| d.as_array())
            .context("embedding reply has no data array")?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let vec: Vec<f32> =
                serde_json::from_value(item.get("embedding").cloned().unwrap_or_default())
                    .context("parse embedding vector")?;
            out.push(vec);
        }
        Ok(out)
    }
}
