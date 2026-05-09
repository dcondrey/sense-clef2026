"""
CES (Constrained Evidence Scoring) subscore computation.

Computes 4 interpretable subscores from V3 features:
  - Coverage: does the answer address the question?
  - Groundedness: is the answer supported by the context?
  - Completeness: is the answer thorough enough?
  - Hallucination: does the answer contradict the context?
"""

import numpy as np


def compute_ces_features(base_features):
    """Compute 4 CES subscores from rubric-track V3 features.

    V3 feature index map:
      7=sim_context, 8=sim_question,
      14=ctx_ans_contradiction, 15=ctx_ans_neutral, 16=ctx_ans_entailment,
      38=min(words/50), 56=entities_ratio
    """
    if base_features.ndim == 1:
        base_features = base_features.reshape(1, -1)

    nli_support = base_features[:, 16]
    nli_contradiction = base_features[:, 14]
    nli_neutral = base_features[:, 15]
    sim_context = base_features[:, 7]
    sim_question = base_features[:, 8]
    answer_len = base_features[:, 38]
    entities_ratio = base_features[:, 56]

    coverage = sim_question
    grounded = np.clip(nli_support - 0.7 * nli_contradiction - 0.3 * nli_neutral, 0, 1)
    completeness = np.clip(0.6 * answer_len + 0.4 * sim_context, 0, 1)
    hallucination = np.clip(0.8 * nli_contradiction + 0.2 * entities_ratio, 0, 1)

    return np.column_stack([coverage, grounded, completeness, hallucination])


def compute_simple_ces_features(base_features):
    """Compute 4 CES subscores from simple-track V3 features.

    Simple track feature indices (42 total):
      0=sim_context, 1=sim_question, 2=sim_q_ctx,
      3=ctx_ans_contra, 4=ctx_ans_neutral, 5=ctx_ans_entail,
      14=min(words/50), 32=entities_ratio
    """
    if base_features.ndim == 1:
        base_features = base_features.reshape(1, -1)

    nli_support = base_features[:, 5]
    nli_contradiction = base_features[:, 3]
    nli_neutral = base_features[:, 4]
    sim_context = base_features[:, 0]
    sim_question = base_features[:, 1]
    answer_len = base_features[:, 14]
    entities_ratio = base_features[:, 32]

    coverage = sim_question
    grounded = np.clip(nli_support - 0.7 * nli_contradiction - 0.3 * nli_neutral, 0, 1)
    completeness = np.clip(0.6 * answer_len + 0.4 * sim_context, 0, 1)
    hallucination = np.clip(0.8 * nli_contradiction + 0.2 * entities_ratio, 0, 1)

    return np.column_stack([coverage, grounded, completeness, hallucination])
