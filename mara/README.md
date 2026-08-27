# mara

A ground-up Rust vector database for local/personal-scale RAG: WAL-durable
storage, hybrid vector + BM25 search (IVF-PQ, LSH, or brute-force exact),
server-side document chunking, an HTTP API, a Rust SDK, and async
leader-follower replication — one daemon process, one on-disk data
directory, no external dependencies (no Postgres, no separate vector
store, no message queue).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how it's built.

## Install

```sh
git clone <this repo> && cd mara
cargo build --release
```

This builds three binaries under `target/release/`:

- **`mara`** — the CLI (`mara-cli`). Talks to a daemon over a Unix socket;
  autostarts one for you if nothing's listening yet (see below).
- **`marad`** — the daemon, for an explicit `systemd`/`launchd`-managed
  install.
- **`mara-api`** — an optional HTTP frontend over the same daemon
  internals, for non-Rust clients.

For everyday use, only `mara` matters — put it on your `$PATH`
(`cargo install --path crates/mara-cli`) and stop there.

## Quickstart

There's no separate "start the server" step. The first `mara` command you
run against a fresh data directory spawns a daemon in the background,
waits for it to become ready, and then runs your command against it —
every command after that reuses the same running daemon:

```sh
$ mara create-collection docs --dim 384
created collection "docs" (dim=384, metric=cosine)

$ mara insert-document docs README.md
inserted document "README.md" (doc_id=0, chunks=17, version=1)

$ mara search docs --text "how does replication work" --mode hybrid -k 3
[
  {
    "row": { "key": "README.md#10", "text": "...", ... },
    "score": 0.031,
    "exact": true
  },
  ...
]
```

`insert-document` splits the file's text into chunks, embeds each one
locally (see *Embedding model* below), and commits the whole document as
one transaction. `search --mode hybrid` fuses vector and BM25 results
server-side via reciprocal rank fusion; `--mode vector` and `--mode bm25`
run just one arm. Everything lands under `~/.mara` by default —
`mara --data-dir <path> ...` or a `mara.toml` in the current directory
(see below) both override that.

Other commands: `mara get <coll> <key>`, `mara delete <coll> <key>`,
`mara put <coll> <key> --vector 1,0,0` (raw-vector insert, no embedding),
`mara reindex <coll> --kind ivf-pq|lsh|bm25|flat`, and
`mara daemon status|stop|logs` to manage the background daemon directly.
`mara repl` opens an interactive shell (one connection reused across
lines, same command grammar); `mara completions zsh|bash|fish` prints a
shell completion script.

### Embedding model

Document chunking needs a configured local embedding model — without one,
`insert-document` and text-mode `search`/`put` fail with a clear
`embedding_not_configured` error rather than silently downloading
something on first use. Set one explicitly in `mara.toml`:

```toml
[server]
data_dir = "/path/to/data"

[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"
```

The model downloads once (via `fastembed`/ONNX Runtime) into
`<data_dir>/models` the first time it's needed.

## HTTP API

`mara-api` is a separate process, run against the same `data_dir`, that
exposes the same operations over REST + JSON instead of the native
socket protocol — the integration point for anything that isn't Rust:

```toml
# mara.toml, in addition to [server]/[embedding] above
[http]
listen = "127.0.0.1:7701"
```

```sh
$ mara-api --config mara.toml &
$ curl -s http://127.0.0.1:7701/v1/health
{"status":"ok"}
$ curl -s -X POST http://127.0.0.1:7701/v1/collections \
    -d '{"name":"docs","dim":384}'
```

The full OpenAPI spec is served at `/v1/openapi.json`; see
[`docs/api-examples.http`](docs/api-examples.http) for a runnable example
of every endpoint. `Authorization: Bearer <token>` resolves to the same
identity/audit path a native socket connection's `Hello` does — see
*Auth* below.

## Rust SDK

`mara-sdk` is a thin wrapper over the pooled native client for Rust
callers that want to skip HTTP entirely:

```rust
let store = mara_sdk::MaraStore::connect_unix("/path/to/mara.sock", "my-app", "docs")?;
store.add_document("readme", &text, None, Default::default(), None).await?;
let hits = store.query("how does replication work", 5).await?;
```

See [`crates/mara-sdk/examples/rag_quickstart.rs`](crates/mara-sdk/examples/rag_quickstart.rs)
for a complete, runnable ingest-then-query example
(`cargo run --example rag_quickstart -p mara-sdk`).

## Auth

Off by default for local single-user use — every local Unix-socket
connection is attributed to the real OS user via `SO_PEERCRED`, so audit
entries always have a genuine subject even unauthenticated. Turn it on
for anything reachable over TCP or shared by multiple principals:

```toml
[auth]
enabled = true
```

Once enabled, every connection (Unix socket, TCP, HTTP, or a replication
follower) needs a bearer token minted against the daemon's token store,
resolving to a `Role` (`Admin`/`Writer`/`Reader`/`Replica`) checked
against one capability table in `mara-daemon::engine`. Token management
isn't wired into the CLI yet (`mara-auth::TokenStore` is the underlying
primitive) — see `docs/ARCHITECTURE.md` for the current state.

## Replication

Leader-follower, async — a follower is the same `marad` binary, pointed
at the leader's own dedicated replication port with a `Role::Replica`
token, replaying the leader's WAL as it streams:

```toml
# leader's mara.toml
[replication]
role = "leader"
listen = "0.0.0.0:7702"
```

```toml
# follower's mara.toml — a separate data_dir/process
[replication]
role = "follower"
leader_addr = "leader-host:7702"
auth_token = "mara_pat_..."
```

A follower serves reads immediately and rejects writes with a clear
`not_the_leader` error; see `docs/ARCHITECTURE.md` for what is and isn't
covered by the current implementation.

## Development

```sh
cargo build --workspace          # build everything
cargo test --workspace           # run the full test suite
cargo clippy --workspace --all-targets
```

`mara-cli`/`mara-daemon`/`mara-embed` link ONNX Runtime by default (the
`embedded` feature, on by default) for local embedding —
`cargo build -p mara-cli --no-default-features` yields a slim,
socket-only client with neither, for a smaller install.
