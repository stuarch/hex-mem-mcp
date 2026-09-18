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
| `memory_store` | `agent_id` (required), `text` (required), `namespace` (=`"default"`), `importance` 0–1 (=0.5), `tags` (=[]) | Embed + store one private memory, returns its numeric `id` |
| `memory_recall` | `agent_id` (required), `query` (required), `namespace` (omit = all), `top_k` 1–50 (=5) | Vector search over this agent's memories only, results in relevance order |
| `memory_forget` | `agent_id` (required), `id` (required) | Delete one memory owned by this agent |
| `memory_stats` | `agent_id` (required), `namespace` (omit = all) | Count this agent's stored memories |

Every tool requires `agent_id`: the calling agent states who it is, and every
query is filtered on that id, so one agent can only ever see its own memories.
`namespace` subdivides one agent's memories per project. Identity is
cooperative — any local caller can claim any id — so this is privacy between
your own agents, not authentication against adversaries.

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
     --embedding --pooling mean
   ```

   Any dense model works; multilingual models (Qwen3-Embedding, …) are
   recommended when memories are not English-only. No API key is needed for a
   local server; if your endpoint requires one, start it with its key option
   and set `HEX_MEM_EMBEDDING_API_KEY` to the same value.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `HEX_MEM_HELIX_URL` | `http://127.0.0.1:6969` | HelixDB base URL |
| `HEX_MEM_EMBEDDING_BASE_URL` | `http://127.0.0.1:51480/v1` | Embeddings API base URL |
| `HEX_MEM_EMBEDDING_MODEL` | *(required)* | Model name sent to the embeddings API |
| `HEX_MEM_EMBEDDING_API_KEY` | *(empty)* | Bearer token, sent only when non-empty |
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

## Upgrading to 0.2.0

`agent_id` became required on every tool, and queries always filter on it.
Memories stored by 0.1.0 have no `agent_id` property, so 0.2.0 can no longer
see them — re-store anything worth keeping.

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
