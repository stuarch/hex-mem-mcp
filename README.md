# hex-mem-mcp

Agent long-term memory MCP server backed by [HelixDB](https://github.com/HelixDB/helix-db).
It exposes four typed tools over stdio so any MCP-capable coding agent
(goose, Claude Code, …) can persist and semantically recall memories.

Memories are stored as `Memory` nodes in HelixDB with a cosine vector index
over the `embedding` property. Embeddings come from any OpenAI-compatible
`/v1/embeddings` endpoint (e.g. a local `llama-server --embedding`).

## Tools

| Tool | Arguments | Effect |
|---|---|---|
| `memory_store` | `text` (required), `namespace` (=`"default"`), `importance` 0–1 (=0.5), `tags` (=[]) | Embed + store one memory, returns its numeric `id` |
| `memory_recall` | `query` (required), `namespace` (omit = all), `top_k` 1–50 (=5) | Vector search, results in relevance order with id/text/namespace/importance/created_at |
| `memory_forget` | `id` (required) | Delete one memory |
| `memory_stats` | `namespace` (omit = all) | Count stored memories |

## Prerequisites

1. A running HelixDB server speaking the v2 HTTP API
   (`GET /healthz`, `POST /v2/query`). The `helix-server` package in the
   [guaix](https://codeberg.org/stuarch/guaix) Guix channel provides a native
   build, no Docker needed:

   ```sh
   HELIX_HTTP_ADDR=127.0.0.1:6969 HELIX_DATA_DIR=/var/lib/helix helix-server
   ```

2. An OpenAI-compatible embeddings endpoint. Example with llama.cpp:

   ```sh
   llama-server -m Qwen3-Embedding-0.6B-Q8_0.gguf --port 51481 \
     --embedding --pooling mean --api-key dummy
   ```

   Any dense model works; multilingual models (Qwen3-Embedding, …) are
   recommended when memories are not English-only.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `HEX_MEM_HELIX_URL` | `http://127.0.0.1:6969` | HelixDB base URL |
| `HEX_MEM_EMBEDDING_BASE_URL` | `http://127.0.0.1:51480/v1` | Embeddings API base URL |
| `HEX_MEM_EMBEDDING_MODEL` | *(required)* | Model name sent to the embeddings API |
| `HEX_MEM_EMBEDDING_API_KEY` | `dummy` | Bearer token, if the endpoint needs one |
| `RUST_LOG` | *(empty)* | e.g. `info` for startup logging on stderr |

On startup the server checks HelixDB health, probes the embedding dimension,
and creates the `Memory.embedding` vector index if needed (index activation
is async server-side; startup waits for it). The embedding model — and
therefore the vector dimension — must stay fixed afterwards.

## Build & run

```sh
cargo build --release
HEX_MEM_EMBEDDING_MODEL=Qwen3-Embedding-0.6B-Q8_0 ./target/release/hex-mem-mcp
```

Requires Rust 1.88+ and a C compiler + cmake (for the rustls TLS backend).

## Agent wiring (goose example)

```yaml
extensions:
  hexmem:
    enabled: true
    type: stdio
    cmd: /path/to/hex-mem-mcp
    args: []
    timeout: 300
    description: Agent long-term memory (HelixDB)
    envs:
      - HEX_MEM_HELIX_URL=http://127.0.0.1:6969
      - HEX_MEM_EMBEDDING_BASE_URL=http://127.0.0.1:51481/v1
      - HEX_MEM_EMBEDDING_MODEL=Qwen3-Embedding-0.6B-Q8_0
```

## Layout

- `src/main.rs` – config, startup checks, index bootstrap, stdio serving
- `src/memory.rs` – the four MCP tools (`#[tool_router]`)
- `src/helix.rs` – HelixDB v2 query client + query JSON builders
- `src/embed.rs` – OpenAI-compatible embeddings client

## License

MIT — see [LICENSE](LICENSE).
