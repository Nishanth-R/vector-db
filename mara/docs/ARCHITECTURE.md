# Architecture

This is a current-state map of how `mara` is actually built, organized by
crate and by request path. For the design history and rationale behind
individual decisions, see `plan/i-am-trying-to-rustling-rabin.md` — this
document describes what exists, not why each choice was made.

## Crate layout

```
mara-proto        wire types: Request/Response, framing codec, Lsn/RowId/DocId,
                   Filter AST, ChunkSpec, RequestCtx — everything else depends on this
mara-auth         Principal/Role/Capability table, TokenStore, AuditSink (JSONL)
mara-storage      Collection, WAL, snapshot/checkpoint, undo/revert, payload store + filters
mara-chunker      text-splitter wrapper: markdown/sentence/token/character chunking
mara-embed        fastembed/ONNX wrapper, model registry, ModelFingerprint
mara-index-vector Flat, IVF, IVF-PQ (+ OPQ), LSH, LiveIndex (delta/tombstone/rebuild)
mara-index-bm25   tokenizer, inverted index, BM25 scoring
mara-fusion       reciprocal-rank and weighted-sum fusion for hybrid search
mara-daemon       Engine (request dispatch + capability checks), listeners,
                   config, replication, composition root (`boot`/`run`)
marad             thin binary wrapping mara-daemon::run
mara-client       pooled async client over the native wire protocol
mara-cli          the `mara` binary: CLI commands, autostart, --embedded
mara-api          axum HTTP frontend over the same Engine, utoipa OpenAPI
mara-sdk          thin single-collection wrapper over mara-client, for Rust RAG apps
```

Dependency direction is strictly downward through that list — `mara-proto`
depends on nothing else in the workspace; `mara-daemon` depends on
`mara-storage`/`mara-index-*`/`mara-auth`/`mara-embed`/`mara-chunker`;
`mara-api` and `mara-cli` both depend on `mara-daemon` (never the reverse
— see *Two frontends, one engine* below for why that matters).

## One request, from wire to disk

1. A client (native socket, HTTP, or an in-process `--embedded` call)
   produces a `mara_proto::Request`.
2. `mara-daemon::engine::EngineImpl::handle` is the single entry point
   every transport calls into. It: checks the request's `Capability`
   against the caller's `Role` (one table, `mara_auth::require`), rejects
   outright if this engine is a replication follower and the request
   mutates rows/collections, dispatches to `mara-storage`/the index
   layers, and always writes exactly one `AuditRecord` — success or
   failure — before returning.
3. A write (`Put`/`PutBatch`/`PutDocument`/`Delete`/`CreateCollection`)
   goes through `mara_storage::Collection`: validate against the payload
   schema, append to the WAL (`WalWriter::append_batch`, assigning each
   record its `Lsn`), apply the same records to in-memory state
   (`recovery::apply_replayed_records` — the *same* function a fresh
   process uses to replay its WAL at startup), then notify
   `ChangeSubscriber`s (`LiveIndex`, BM25's live index) with the derived
   `ChangeEvent`s.
4. A search dispatches to whichever `VectorIndex` is cached for that
   collection (`FlatIndex` if nothing's been built yet — it re-scans live
   storage every call, so it's always correct, just not fast at scale)
   and/or the BM25 index, then — for `Hybrid` mode — fuses both arms via
   `mara-fusion` before grouping by `max_chunks_per_doc`.

## Storage: WAL, snapshots, undo

Each collection is a directory (`<data_dir>/collections/<name>/`) with its
own `wal/segment-*.wal` files and `snapshot-*.mdb` checkpoints — never
shared across collections. The WAL is JSONL, one record per line, each
carrying a checksum (`crc32c` over the record with its own checksum field
blanked) and an `(segment_id, byte_offset)` `Lsn`. A crash can only ever
corrupt the *last* line; `wal::replay_all` reads until the first
checksum failure or incomplete trailing transaction and reports exactly
how far the file can be trusted, and `Collection::open` truncates to that
point before resuming writes — this *is* crash recovery, not a separate
mode.

Undo (`Collection::undo`/`revert_to`) is a compensating write, not a
special record type: reversing a transaction means replaying its
`WalRecord::undo` payloads as a brand-new forward-appended transaction,
tagged `caused_by` the transaction it reverses. Nothing about undo needs
to know it's undo at apply time, which is also exactly why replication
needs zero special-casing for it (see below).

Snapshots (`mara_storage::snapshot`) are periodic, bincode-encoded,
checksummed checkpoints of a collection's full in-memory state, tagged
with the `Lsn` they're consistent with — `Collection::open` loads the
newest valid one, then replays only the WAL tail after it, rather than
replaying from empty every restart.

## Vector indexing

`FlatIndex` (exact, brute-force) is never a fallback bolted on last — it's
the recall oracle every approximate index is validated against, and the
default until `Reindex` builds something else. `IvfPqIndex` adds coarse
k-means clustering (optionally hierarchical, with beam search) plus
product-quantized codes and asymmetric distance computation for the
shortlist, then exact-reranks candidates against raw vectors before
returning scores — so a returned score is always the true similarity,
never a quantization estimate, regardless of which index answered.
`LshIndex` (SimHash, multiple hash tables) needs no training and exists
both as a zero-warmup alternative and a recall cross-check against
IVF-PQ.

Filtered search runs one of three regimes depending on selectivity,
chosen per query: exact-scan below `filter_exact_threshold`, filtered-ANN
with dynamic `nprobe` escalation in the middle band, and plain ANN with a
post-hoc mask above `filter_selectivity_high` — `truncated_by_filter`
reports whether an escalation hit its candidate-budget cap short of a
full `k`, distinguished from the collection genuinely having fewer than
`k` matches.

`LiveIndex` wraps a "baked" index (built from a full scan at some point in
time) with a `delta`/`tombstones` overlay fed by `ChangeSubscriber`, so a
write is searchable immediately without waiting for the next `Reindex` —
`should_rebuild()`/`rebuild()` are triggered opportunistically after
writes, off the request path via `spawn_blocking`, not a periodic timer.

## Replication

Leader-follower, async — the leader's own local WAL+fsync is the real
durability boundary; there is no ack/quorum tracking. A follower is the
same `marad` binary with `[replication] role = "follower"`: it connects
to the leader's *dedicated* replication TCP listener (separate from the
normal client-facing UDS/TCP ports) with a `Role::Replica` token, sends
`ReplicaHello{known_lsns}` (its resume cursor per collection, `None`
meaning "from the very first record" — `Lsn::ZERO` is itself a valid LSN,
not a sentinel, so this distinction matters), and receives a
`ReplicaWelcome` listing every collection to mirror locally, followed by
a continuous, unprompted stream of `ReplicaWalLines`/`ReplicaNewCollection`
frames — no further request needed once the stream starts.

Because replicated records are literally WAL lines, a follower's apply
path (`Collection::apply_replicated_lines`) both parses/checksum-verifies
them and appends them to its own local WAL *preserving the leader's exact
LSNs* (`WalWriter::append_replicated_batch`, which lands each record at
its own LSN's segment/offset rather than minting a fresh one) before
applying them to in-memory state through the same recovery path a normal
restart uses. A follower's `EngineImpl` rejects row/collection-mutating
requests with `not_the_leader`; `Reindex` is deliberately still allowed,
since it only ever rebuilds *derived* index state from rows the follower
already has, the same way it never touches storage's own durable state on
the leader either.

**What's not implemented yet**: snapshot-bootstrap for a follower that's
very far behind (WAL segments are never pruned by this codebase, so a
WAL-tail replay from the first record always works, just less efficiently
than a snapshot transfer would over a very long history); and `Reindex`
WAL-record propagation, so a follower doesn't yet retrain its derived
indexes automatically on the same trigger the leader did — the current
`Reindex` handler doesn't append a WAL record at all, leader or follower,
so there's nothing yet for a follower to react to. Building an index on a
follower today means calling `Reindex` against it directly.

## Two frontends, one engine

`mara-cli` and `mara-api` are peer processes, not a client/server pair —
both depend on `mara-daemon` directly and both call
`mara_daemon::server::boot` to construct the exact same
`Storage`/`Engine`/`TokenStore`/audit-sink composition, so a bearer token
resolved over HTTP and a `Hello.auth_token` resolved over a Unix socket
produce identical `RequestCtx`s, hit the identical capability table, and
land in the identical audit log. Neither can be a dependency of the
other — `mara-daemon` has no HTTP awareness, and `mara-api` never touches
raw sockets — so they're deliberately separate OS processes against the
same `data_dir`, not one mounted inside the other. `mara-sdk` skips both:
it's a thin wrapper over `mara-client`, the same pooled native-socket
client `mara-cli` uses, for Rust callers who want to avoid the HTTP hop
entirely.

## Auth and audit

`Role` (`Admin`/`Writer`/`Reader`/`Replica`) is resolved once per
connection (or, for HTTP, per request) from either a bearer token
(`mara_auth::TokenStore`, blake3-hashed, never storing the plaintext) or,
when `[auth] enabled = false`, a synthetic principal derived from the
connection's `Source` — a Unix-socket connection always resolves to the
real OS user via `SO_PEERCRED`, so the audit log has a genuine subject
even with auth off entirely. Every `(Role, Capability)` pair allowed is
listed in one function (`mara_auth::capability::require`) rather than
scattered through the codebase, so the whole authorization surface is
auditable by reading one match statement. Every request produces exactly
one `AuditRecord` (JSONL, rotated daily and by size) — the audit log
itself is never replicated; each node audits only what it actually
served.
