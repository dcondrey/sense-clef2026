"""Smoke tests: verify all sense modules import and key classes exist."""

import logging

log = logging.getLogger(__name__)


def test_import_sense_package():
    """Top-level sense package imports without error."""
    import sense
    assert hasattr(sense, "__version__")
    log.info("sense package version: %s", sense.__version__)


def test_import_ces_module():
    """sense.ces module imports and exposes expected functions."""
    from sense import ces
    assert callable(ces.compute_ces_features)
    assert callable(ces.compute_simple_ces_features)


def test_import_features_module():
    """sense.features module imports and exposes FeatureExtractor."""
    from sense.features import FeatureExtractor
    assert callable(FeatureExtractor)


def test_import_ensemble_module():
    """sense.ensemble module imports and exposes KDE/scoring helpers."""
    from sense.ensemble import (
        score_to_probs_kde,
        score_to_probs_kde_perclass,
        safe_probs,
        retrieval_score,
        get_question_key,
    )
    assert callable(score_to_probs_kde)
    assert callable(score_to_probs_kde_perclass)
    assert callable(safe_probs)
    assert callable(retrieval_score)
    assert callable(get_question_key)


def test_import_translate_module():
    """sense.translate module imports and exposes translation helpers."""
    from sense.translate import detect_irish, translate_items
    assert callable(detect_irish)
    assert callable(translate_items)


def test_import_io_module():
    """sense.io module imports and exposes I/O functions."""
    from sense.io import load_input_data, write_output, detect_track, load_models
    assert callable(load_input_data)
    assert callable(write_output)
    assert callable(detect_track)
    assert callable(load_models)


def test_import_models_module():
    """sense.models module imports and exposes model classes."""
    from sense.models import (
        CORALHead,
        DeBERTaRubricScorer,
        RubricDataset,
        SimpleDataset,
        dynamic_collate_fn,
    )
    assert callable(CORALHead)
    assert callable(DeBERTaRubricScorer)
    assert callable(RubricDataset)
    assert callable(SimpleDataset)
    assert callable(dynamic_collate_fn)


def test_feature_extractor_signature():
    """FeatureExtractor.__init__ accepts sbert_model and nli_model."""
    import inspect
    from sense.features import FeatureExtractor

    sig = inspect.signature(FeatureExtractor.__init__)
    params = list(sig.parameters.keys())
    assert "sbert_model" in params, f"Expected 'sbert_model' in {params}"
    assert "nli_model" in params, f"Expected 'nli_model' in {params}"


def test_ces_functions_exist():
    """compute_ces_features and compute_simple_ces_features are importable from top level."""
    from sense import compute_ces_features, compute_simple_ces_features
    assert callable(compute_ces_features)
    assert callable(compute_simple_ces_features)
