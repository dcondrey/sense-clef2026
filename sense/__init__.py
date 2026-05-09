"""
SENSE: Sensemaking Evaluation via NLI and Semantic Entailment.

PAN@CLEF 2026 ELOQUENT Sensemaking Task.
"""

__version__ = "1.0.0"

from sense.models import CORALHead, DeBERTaRubricScorer, RubricDataset, SimpleDataset, dynamic_collate_fn
from sense.features import FeatureExtractor
from sense.ces import compute_ces_features, compute_simple_ces_features
from sense.ensemble import (
    score_to_probs_kde,
    score_to_probs_kde_perclass,
    safe_probs,
    retrieval_score,
    get_question_key,
)
from sense.translate import detect_irish, translate_items
from sense.io import load_input_data, write_output, detect_track, load_models

__all__ = [
    "__version__",
    "CORALHead",
    "DeBERTaRubricScorer",
    "RubricDataset",
    "SimpleDataset",
    "dynamic_collate_fn",
    "FeatureExtractor",
    "compute_ces_features",
    "compute_simple_ces_features",
    "score_to_probs_kde",
    "score_to_probs_kde_perclass",
    "safe_probs",
    "retrieval_score",
    "get_question_key",
    "detect_irish",
    "translate_items",
    "load_input_data",
    "write_output",
    "detect_track",
    "load_models",
]
