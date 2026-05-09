"""Tests for CES (Constrained Evidence Scoring) subscore computation."""

import logging

import numpy as np

from sense.ces import compute_ces_features, compute_simple_ces_features

log = logging.getLogger(__name__)


def test_compute_ces_features_shape_1d():
    """compute_ces_features produces (1, 4) output from a 73-dim vector."""
    features = np.random.default_rng(42).random(73)
    result = compute_ces_features(features)
    assert result.shape == (1, 4), f"Expected (1, 4), got {result.shape}"
    log.info("CES 1-D output shape: %s", result.shape)


def test_compute_ces_features_shape_2d():
    """compute_ces_features produces (N, 4) output from an (N, 73) matrix."""
    rng = np.random.default_rng(42)
    features = rng.random((10, 73))
    result = compute_ces_features(features)
    assert result.shape == (10, 4), f"Expected (10, 4), got {result.shape}"


def test_compute_ces_features_values_bounded():
    """CES subscores are clipped to [0, 1] except coverage (raw sim_question)."""
    rng = np.random.default_rng(42)
    features = rng.random(73)
    result = compute_ces_features(features)
    # grounded, completeness, hallucination are clipped
    for col_idx, name in [(1, "grounded"), (2, "completeness"), (3, "hallucination")]:
        val = result[0, col_idx]
        assert 0.0 <= val <= 1.0, f"{name} = {val} is out of [0, 1]"


def test_compute_simple_ces_features_shape():
    """compute_simple_ces_features produces (1, 4) from a 42-dim vector."""
    features = np.random.default_rng(42).random(42)
    result = compute_simple_ces_features(features)
    assert result.shape == (1, 4), f"Expected (1, 4), got {result.shape}"


def test_compute_ces_features_deterministic():
    """Same input produces same output."""
    rng = np.random.default_rng(42)
    features = rng.random(73)
    result1 = compute_ces_features(features)
    result2 = compute_ces_features(features)
    np.testing.assert_array_equal(result1, result2)


def test_compute_ces_features_batch_equals_single():
    """Batched computation matches individual computation."""
    rng = np.random.default_rng(42)
    batch = rng.random((5, 73))
    batch_result = compute_ces_features(batch)
    for i in range(5):
        single_result = compute_ces_features(batch[i])
        np.testing.assert_allclose(
            batch_result[i], single_result[0],
            atol=1e-10,
            err_msg=f"Mismatch at row {i}",
        )
