//! Canonical model-id resolution against `fastembed`'s supported models.
//!
//! `fastembed::ModelInfo::model_code` is the *ONNX-repackaged* Hugging Face
//! repo it downloads from (e.g. `"Xenova/all-MiniLM-L6-v2"`), not the
//! canonical model name a user would type into config
//! (`"sentence-transformers/all-MiniLM-L6-v2"`) — the two are shown as a
//! doc comment on each `fastembed::EmbeddingModel` variant, not exposed as
//! data, so this table is hand-maintained against those doc comments
//! rather than derived mechanically. What *is* derived mechanically is the
//! dimension: `TextEmbedding::get_model_info` returns fastembed's own
//! static `ModelInfo`, so `dim` can never drift out of sync with the
//! model fastembed actually loads.
//!
//! Resolution here never downloads or loads anything — `ModelInfo` is
//! compiled-in metadata — so it works fully offline.

use crate::error::{EmbedError, EmbedResult};
use fastembed::{EmbeddingModel, TextEmbedding};
use mara_proto::ModelFingerprint;

const CANONICAL_MODELS: &[(&str, EmbeddingModel)] = &[
    ("sentence-transformers/all-MiniLM-L6-v2", EmbeddingModel::AllMiniLML6V2),
    ("sentence-transformers/all-MiniLM-L12-v2", EmbeddingModel::AllMiniLML12V2),
    ("sentence-transformers/all-mpnet-base-v2", EmbeddingModel::AllMpnetBaseV2),
    ("sentence-transformers/paraphrase-mpnet-base-v2", EmbeddingModel::ParaphraseMLMpnetBaseV2),
    ("BAAI/bge-small-en-v1.5", EmbeddingModel::BGESmallENV15),
    ("BAAI/bge-base-en-v1.5", EmbeddingModel::BGEBaseENV15),
    ("BAAI/bge-large-en-v1.5", EmbeddingModel::BGELargeENV15),
    ("BAAI/bge-small-zh-v1.5", EmbeddingModel::BGESmallZHV15),
    ("BAAI/bge-large-zh-v1.5", EmbeddingModel::BGELargeZHV15),
    ("BAAI/bge-m3", EmbeddingModel::BGEM3),
    ("nomic-ai/nomic-embed-text-v1", EmbeddingModel::NomicEmbedTextV1),
    ("nomic-ai/nomic-embed-text-v1.5", EmbeddingModel::NomicEmbedTextV15),
    ("intfloat/multilingual-e5-small", EmbeddingModel::MultilingualE5Small),
    ("intfloat/multilingual-e5-base", EmbeddingModel::MultilingualE5Base),
    ("intfloat/multilingual-e5-large", EmbeddingModel::MultilingualE5Large),
    ("mixedbread-ai/mxbai-embed-large-v1", EmbeddingModel::MxbaiEmbedLargeV1),
    ("Alibaba-NLP/gte-base-en-v1.5", EmbeddingModel::GTEBaseENV15),
    ("Alibaba-NLP/gte-large-en-v1.5", EmbeddingModel::GTELargeENV15),
];

pub fn known_model_ids() -> Vec<&'static str> {
    CANONICAL_MODELS.iter().map(|(id, _)| *id).collect()
}

pub(crate) fn variant_for(model_id: &str) -> Option<EmbeddingModel> {
    CANONICAL_MODELS.iter().find(|(id, _)| *id == model_id).map(|(_, v)| v.clone())
}

/// Resolves a configured model string to a `ModelFingerprint`. An unknown
/// string produces an error listing supported ids, per the master plan —
/// never a runtime failure deferred to first insert.
pub fn resolve_model(spec: &str) -> EmbedResult<ModelFingerprint> {
    if let Some(path) = spec.strip_prefix("custom:") {
        return Err(EmbedError::CustomModelUnsupported(path.to_string()));
    }
    let variant = variant_for(spec).ok_or_else(|| EmbedError::UnknownModel {
        model: spec.to_string(),
        known: known_model_ids().join(", "),
    })?;
    let info = TextEmbedding::get_model_info(&variant).map_err(|e| EmbedError::Backend(e.to_string()))?;
    Ok(ModelFingerprint {
        model_id: spec.to_string(),
        revision: None,
        dim: info.dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_known_models_with_correct_dims_and_no_network() {
        let fp = resolve_model("sentence-transformers/all-MiniLM-L6-v2").unwrap();
        assert_eq!(fp.model_id, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(fp.dim, 384);

        let fp = resolve_model("BAAI/bge-base-en-v1.5").unwrap();
        assert_eq!(fp.dim, 768);
    }

    #[test]
    fn every_registry_entry_resolves_via_fastembeds_own_metadata() {
        // Guards against the table drifting out of sync with fastembed's
        // enum: every canonical id we claim to support must actually
        // resolve to a real ModelInfo.
        for (id, _) in CANONICAL_MODELS {
            resolve_model(id).unwrap_or_else(|e| panic!("registry entry {id:?} failed to resolve: {e}"));
        }
    }

    #[test]
    fn unknown_model_lists_supported_ids_in_the_error() {
        let err = resolve_model("not-a-real-model").unwrap_err();
        match err {
            EmbedError::UnknownModel { model, known } => {
                assert_eq!(model, "not-a-real-model");
                assert!(known.contains("sentence-transformers/all-MiniLM-L6-v2"));
            }
            other => panic!("expected UnknownModel, got {other:?}"),
        }
    }

    #[test]
    fn custom_model_prefix_is_a_clear_not_yet_supported_error() {
        let err = resolve_model("custom:/path/to/onnx-dir").unwrap_err();
        assert!(matches!(err, EmbedError::CustomModelUnsupported(p) if p == "/path/to/onnx-dir"));
    }
}
