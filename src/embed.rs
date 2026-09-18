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

    /// Build the embeddings POST request. An empty API key (the default)
    /// sends no Authorization header, for endpoints that need no auth.
    fn build_request(&self, inputs: &[String]) -> Result<reqwest::Request> {
        let url = format!("{}/embeddings", self.base_url);
        let builder = self
            .http
            .post(&url)
            .json(&json!({"model": self.model, "input": inputs}));
        let builder = if self.api_key.is_empty() {
            builder
        } else {
            builder.bearer_auth(&self.api_key)
        };
        builder.build().context("build embedding request")
    }

    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let req = self.build_request(inputs)?;
        let resp = self
            .http
            .execute(req)
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

#[cfg(test)]
mod tests {
    use super::Embedder;

    fn embedder_with_key(key: &str) -> Embedder {
        Embedder::new("http://127.0.0.1:9/v1", "model", key).unwrap()
    }

    #[test]
    fn empty_key_sends_no_authorization_header() {
        let req = embedder_with_key("")
            .build_request(&["hi".to_string()])
            .unwrap();
        assert_eq!(req.url().as_str(), "http://127.0.0.1:9/v1/embeddings");
        assert!(
            !req.headers().contains_key(reqwest::header::AUTHORIZATION),
            "empty key must omit the header, got: {:?}",
            req.headers()
        );
    }

    #[test]
    fn nonempty_key_sends_bearer_header() {
        let req = embedder_with_key("s3cret")
            .build_request(&["hi".to_string()])
            .unwrap();
        let auth = req.headers()[reqwest::header::AUTHORIZATION]
            .to_str()
            .unwrap();
        assert_eq!(auth, "Bearer s3cret");
    }
}
