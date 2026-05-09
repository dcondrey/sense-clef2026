"""
Ensemble combination via KDE Product of Experts (PoE).

Provides KDE-based probability estimation, retrieval scoring,
and threshold optimization for the final prediction stage.
"""

import numpy as np
from scipy.stats import norm
from collections import defaultdict


def score_to_probs_kde(scores, train_scores, train_labels, bandwidth=0.23, n_classes=3):
    """Vectorized KDE probability estimation."""
    probs = np.zeros((len(scores), n_classes))
    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() == 0:
            continue
        class_scores = train_scores[mask]
        diffs = (scores[:, None] - class_scores[None, :]) / bandwidth
        probs[:, c] = np.mean(norm.pdf(diffs), axis=1)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
    return probs


def score_to_probs_kde_perclass(scores, train_scores, train_labels, bandwidths, n_classes=None):
    """Vectorized KDE with per-class bandwidth."""
    if n_classes is None:
        n_classes = len(bandwidths)
    probs = np.zeros((len(scores), n_classes))
    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() == 0:
            continue
        class_scores = train_scores[mask]
        bw = bandwidths[c]
        diffs = (scores[:, None] - class_scores[None, :]) / bw
        probs[:, c] = np.mean(norm.pdf(diffs), axis=1)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
    return probs


def safe_probs(p, eps=1e-8):
    """Clip and renormalize probabilities."""
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def get_question_key(item):
    """Extract question key for retrieval grouping."""
    return item.get('input', item).get('rubrics', {}).get('FC', '')[:100]


def retrieval_score(query_items, query_emb, pool_items, pool_emb, pool_labels,
                    sigma=0.1, exclude_self=False):
    """Compute retrieval-based class probabilities."""
    n = len(query_items)
    scores = np.zeros((n, 3))
    raw_scores = np.zeros(n)
    q_to_pool = defaultdict(list)
    for i, item in enumerate(pool_items):
        q_to_pool[get_question_key(item)].append(i)
    for i, item in enumerate(query_items):
        qkey = get_question_key(item)
        neighbors = q_to_pool.get(qkey, [])
        if exclude_self:
            neighbors = [j for j in neighbors if j != i]
        if len(neighbors) > 0:
            sims = np.dot(pool_emb[neighbors], query_emb[i])
            weights = np.exp(sims / sigma)
            weights /= weights.sum()
            for c in range(3):
                class_mask = pool_labels[neighbors] == c
                scores[i, c] = weights[class_mask].sum()
            raw_scores[i] = np.sum(weights * pool_labels[neighbors])
        else:
            scores[i] = [0.28, 0.28, 0.44]
            raw_scores[i] = 1.0
    return scores, raw_scores
