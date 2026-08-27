//! Shared between the socket-client path and (with the `embedded` feature)
//! the in-process path: turning a parsed CLI command into a `Request`, and
//! a `Response` back into what the user sees. Keeping this one place is
//! what guarantees the two execution paths behave identically.

use crate::Commands;
use mara_proto::{ChunkSpec, ChunkStrategy, PayloadRow, Request, Response, SearchMode, WireFieldType, WireIndexKind, WireSearchParams};

pub fn parse_metric(s: &str) -> Result<mara_proto::DistanceMetric, String> {
    match s.to_ascii_lowercase().as_str() {
        "cosine" => Ok(mara_proto::DistanceMetric::Cosine),
        "l2" | "euclidean" => Ok(mara_proto::DistanceMetric::L2),
        "dot" | "dotproduct" | "dot_product" => Ok(mara_proto::DistanceMetric::DotProduct),
        other => Err(format!("unknown metric {other:?}; expected cosine, l2, or dot")),
    }
}

pub fn parse_vector(s: &str) -> Result<Vec<f32>, String> {
    s.split(',')
        .map(|part| part.trim().parse::<f32>().map_err(|e| format!("invalid vector component {part:?}: {e}")))
        .collect()
}

/// `characters | markdown | sentences | tokens:<tokenizer-id>` — the
/// `--chunk-strategy` flag's grammar. `max_tokens`/`overlap_tokens`
/// override `ChunkSpec::default()`'s corresponding fields when given;
/// `--chunk-strategy` alone (no overrides) is the common case.
pub fn parse_chunk_spec(strategy: &str, max_tokens: Option<usize>, overlap_tokens: Option<usize>) -> Result<ChunkSpec, String> {
    let default = ChunkSpec::default();
    let strategy = match strategy {
        "characters" => ChunkStrategy::Characters,
        "sentences" => ChunkStrategy::Sentences,
        "markdown" => ChunkStrategy::Markdown { respect_headings: true },
        other => match other.strip_prefix("tokens:") {
            Some(tokenizer) if !tokenizer.is_empty() => ChunkStrategy::Tokens { tokenizer: tokenizer.to_string() },
            _ => return Err(format!("unknown --chunk-strategy {other:?}; expected characters, markdown, sentences, or tokens:<tokenizer-id>")),
        },
    };
    Ok(ChunkSpec {
        strategy,
        max_tokens: max_tokens.unwrap_or(default.max_tokens),
        overlap_tokens: overlap_tokens.unwrap_or(default.overlap_tokens),
        trim: default.trim,
    })
}

/// `None` for commands that aren't storage requests at all (`serve`,
/// `daemon ...`) — those are handled entirely by the CLI itself.
pub fn command_to_request(cmd: &Commands) -> Result<Option<Request>, String> {
    let req = match cmd {
        Commands::CreateCollection { name, dim, metric } => Request::CreateCollection {
            name: name.clone(),
            dim: *dim,
            metric: parse_metric(metric)?,
            schema: Vec::<(String, WireFieldType)>::new(),
        },
        Commands::Put { coll, key, vector } => Request::Put {
            coll: coll.clone(),
            key: key.clone(),
            text: None,
            vector: Some(parse_vector(vector)?),
            fields: PayloadRow::new(),
            extra: None,
        },
        Commands::Get { coll, key } => Request::GetByKey { coll: coll.clone(), key: key.clone() },
        Commands::Delete { coll, key } => Request::Delete { coll: coll.clone(), key: key.clone() },
        Commands::InsertDocument {
            coll,
            path,
            doc_key,
            chunk_strategy,
            max_tokens,
            overlap_tokens,
        } => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("failed to read {path:?}: {e}"))?;
            let doc_key = doc_key.clone().unwrap_or_else(|| {
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned())
            });
            Request::PutDocument {
                coll: coll.clone(),
                doc_key,
                text,
                chunk_spec: parse_chunk_spec(chunk_strategy, *max_tokens, *overlap_tokens)?,
                fields: PayloadRow::new(),
                source: Some(path.to_string_lossy().into_owned()),
            }
        }
        Commands::Search { coll, vector, text, mode, k } => {
            let mode = match mode.to_ascii_lowercase().as_str() {
                "vector" => SearchMode::VectorOnly,
                "bm25" => SearchMode::Bm25Only,
                "hybrid" => SearchMode::Hybrid { method: None, overfetch_k: None },
                other => return Err(format!("unknown --mode {other:?}; expected vector, bm25, or hybrid")),
            };
            let query_vector = vector.as_deref().map(parse_vector).transpose()?;
            if query_vector.is_none() && text.is_none() {
                return Err("search requires --vector or --text".into());
            }
            Request::Search {
                coll: coll.clone(),
                query_text: text.clone(),
                query_vector,
                mode,
                k: *k,
                filter: None,
                params: WireSearchParams::default(),
            }
        }
        Commands::Reindex { coll, kind } => {
            let index_kind = match kind.to_ascii_lowercase().as_str() {
                "flat" => WireIndexKind::Flat,
                "ivf-pq" | "ivf_pq" | "ivfpq" => WireIndexKind::IvfPq,
                "lsh" => WireIndexKind::Lsh,
                "bm25" => WireIndexKind::Bm25,
                other => return Err(format!("unknown --kind {other:?}; expected flat, ivf-pq, lsh, or bm25")),
            };
            Request::Reindex { coll: coll.clone(), index_kind }
        }
        #[cfg(feature = "embedded")]
        Commands::Serve { .. } => return Ok(None),
        Commands::Daemon { .. } => return Ok(None),
        Commands::Completions { .. } => return Ok(None),
        Commands::Repl => return Ok(None),
    };
    Ok(Some(req))
}

/// Prints `response` and turns a server-side error into a CLI error
/// (non-zero exit, per `main`'s `Err(String)` convention) rather than
/// printing it and exiting 0.
pub fn print_response(cmd: &Commands, response: Response) -> Result<(), String> {
    match response {
        Response::Ok => {
            match cmd {
                Commands::CreateCollection { name, dim, metric } => println!("created collection {name:?} (dim={dim}, metric={metric})"),
                Commands::Delete { coll, key } => println!("deleted {key:?} from {coll:?}"),
                _ => println!("ok"),
            }
            Ok(())
        }
        Response::Row(row) => {
            println!("{}", serde_json::to_string_pretty(&row).expect("Row always serializes"));
            Ok(())
        }
        Response::Rows(rows) => {
            println!("{}", serde_json::to_string_pretty(&rows).expect("rows always serialize"));
            Ok(())
        }
        Response::Document { doc_id, doc_key, chunk_count, version } => {
            println!("inserted document {doc_key:?} (doc_id={doc_id}, chunks={chunk_count}, version={version})");
            Ok(())
        }
        Response::SearchResults { hits, truncated_by_filter } => {
            println!("{}", serde_json::to_string_pretty(&hits).expect("hits always serialize"));
            if truncated_by_filter {
                eprintln!("warning: results may be incomplete — the filtered search hit its escalation budget before finding a full k");
            }
            Ok(())
        }
        Response::Reindexed { coll, index_kind, row_count } => {
            println!("reindexed {coll:?} ({index_kind:?}, {row_count} rows)");
            Ok(())
        }
        Response::HelloAck { .. } => Ok(()),
        // The replication sub-protocol's own responses — `mara-cli` never
        // sends `ReplicaHello`, so a real daemon never sends these back to
        // it; unreachable in practice, kept as a clear error rather than a
        // panic if that assumption is ever violated.
        Response::ReplicaWelcome { .. } | Response::ReplicaNewCollection { .. } | Response::ReplicaWalLines { .. } => {
            Err("unexpected replication-protocol response on a normal client connection".into())
        }
        Response::Error { code, message } => {
            if code == "not_found" {
                if let Commands::Get { coll, key } = cmd {
                    return Err(format!("not found: {key:?} in {coll:?}"));
                }
            }
            Err(format!("[{code}] {message}"))
        }
    }
}
