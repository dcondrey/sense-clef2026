# SENSE: Sensemaking Evaluation via NLI and Semantic Entailment

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)
![CLEF 2026](https://img.shields.io/badge/CLEF-2026-orange)
![Task](https://img.shields.io/badge/task-ELOQUENT%20Sensemaking-purple)

**Neuro-symbolic rubric verification with constrained evidence scoring for multilingual answer assessment.**

Team `writerslogic` submission to the [PAN@CLEF 2026 ELOQUENT Sensemaking Task](https://pan.webis.de/clef26/pan26-web/sensemaking.html), scoring **QWK 0.990** on the rubric track (59/4146 errors, 1.42% error rate).

---

## Key Results

| Track | Metric | Score |
|-------|--------|-------|
| Rubric (3-class) | Quadratic Weighted Kappa | **0.990** |
| Rubric (3-class) | Accuracy | 98.6% |
| Simple (5-class) | Quadratic Weighted Kappa | 0.914 |

### Development Trajectory

| Phase | Approach | QWK | Key Insight |
|-------|----------|-----|-------------|
| Phase 0 | Best heuristic (embedding differentials) | 0.362 | Ceiling without supervision |
| Phase 1 | ML V1 (SBERT + NLI + XGBoost) | ~0.72 | Supervised learning unlocks signal |
| Phase 2 | CES (81 features, monotonic LGB) | 0.868 | Constrained evidence scoring |
| Phase 3 | + DeBERTa CORAL cross-encoder | 0.914 | Ordinal regression captures rubric semantics |
| Phase 4 | + KDE PoE ensemble | 0.970 | Product of Experts calibration |
| Phase 5 | + Irish translation + noise-aware training | 0.990 | Per-language calibration, cleanlab weighting |

---

## Architecture

The system runs a 6-stage pipeline on each input:

```mermaid
graph TD
    A[Input: context, question, answer, rubrics] --> B[Irish Translation]
    B --> C[V3 Feature Extraction]
    C --> D[CES Subscores]
    C --> E[DeBERTa CORAL]
    D --> F[LGB + MLP Ensemble]
    E --> G[KDE Product of Experts]
    F --> G
    G --> H[Predicted Label]

    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style G fill:#fff3e0
```

1. **Irish Translation**: Detects Irish (Gaeilge) answers via keyword matching and translates with `Helsinki-NLP/opus-mt-ga-en`. Other languages are handled natively by multilingual models.
2. **V3 Feature Extraction**: 73 features across 4 groups using SBERT embeddings and NLI cross-encoder scores.
3. **CES Subscores**: 4 interpretable evidence-grounded subscores with monotonic constraints.
4. **DeBERTa CORAL**: Fine-tuned `cross-encoder/nli-deberta-v3-base` with ordinal CORAL regression head.
5. **LGB + MLP Ensemble**: LightGBM (monotonic-constrained) and MLP regressors averaged, 5-seed ensembles.
6. **KDE PoE**: Product of Experts combination using per-class kernel density estimation.

---

## Why This Works

### 1. Constrained Evidence Scoring (CES)

The 4 CES subscores enforce domain-knowledge constraints via LightGBM monotonic features:

| Subscore | Direction | Formula |
|----------|-----------|---------|
| Coverage | +1 (higher = better) | `sim(answer, question)` |
| Groundedness | +1 (higher = better) | `NLI_entail - 0.7*NLI_contra - 0.3*NLI_neutral` |
| Completeness | +1 (higher = better) | `0.6*answer_length + 0.4*sim(answer, context)` |
| Hallucination | -1 (higher = worse) | `0.8*NLI_contra + 0.2*entity_ratio` |

Monotonic constraints prevent the model from learning spurious shortcuts (e.g., "high hallucination = high score").

### 2. Ordinal Regression via CORAL

The task is ordinal: NC < PC < FC (or 0 < 1 < 2 < 3 < 4). Standard classification ignores this structure. CORAL (Consistent Rank Logits) models cumulative probabilities P(Y > k), producing calibrated ordinal predictions that respect the label ordering. Combined with a differentiable soft-QWK loss term, this directly optimizes the evaluation metric.

### 3. KDE Product of Experts

Instead of simple averaging, SENSE converts each model's continuous scores into class probabilities via kernel density estimation (with per-class bandwidths optimized on OOF predictions), then combines them as a Product of Experts in log-probability space. This produces sharper posteriors than arithmetic averaging, especially for boundary cases.

---

## Quick Start

### Install

```bash
pip install -e .
```

### Train

```bash
python train.py --data path/to/dev.rubric.json --output models/
```

### Predict (TIRA)

```bash
python predict.py -i /path/to/input -o /path/to/output
```

### Evaluate

```bash
python evaluate.py -d path/to/devset/ -p predictions/ -t rubric
```

### Docker (TIRA submission)

```bash
# Train models first
python train.py --data dev.rubric.json --output models/

# Build and run
docker build -t sense-clef2026 .
docker run --rm -v /input:/input -v /output:/output sense-clef2026 -i /input -o /output
```

---

## Feature Engineering

### V3 Features (73 dimensions, rubric track)

| Group | Count | Description |
|-------|-------|-------------|
| **Embedding Similarity** | 14 | Cosine similarity between answer, FC/PC/NC rubrics, context, question embeddings; pairwise differentials; dominance score |
| **NLI Scores** | 22 | Bidirectional entailment/contradiction/neutral probabilities for (answer, rubric), (answer, context) pairs; differentials |
| **Structural** | 22 | Word/char counts, sentence count, punctuation density, named entities, vocabulary diversity, length thresholds |
| **Compression** | 7 | Normalized compression distance (NCD) to rubrics and context; self-compression ratio |
| **Rubric Keywords** | 4 | Keyword overlap with FC/NC rubric text |
| **Cross-lingual** | 4 | ASCII detection, Czech/German/Cyrillic character flags |

Models: `paraphrase-multilingual-MiniLM-L12-v2` (SBERT), `cross-encoder/nli-deberta-v3-base` (NLI).

### CES Subscores (4 dimensions)

Prepended to V3 features as the first 4 columns. The LightGBM monotonic constraints `[+1, +1, +1, -1, 0, 0, ..., 0]` ensure Coverage, Groundedness, and Completeness increase the predicted score while Hallucination decreases it.

---

## Ablation Study

Impact of each component, measured on the full 4,146-item devset:

| Configuration | QWK | Delta |
|---------------|-----|-------|
| SBERT similarity only (no ML) | 0.362 | baseline |
| CES 81 features + LGB | 0.868 | +0.506 |
| + DeBERTa CORAL (single model) | 0.914 | +0.046 |
| + 5-seed LGB + MLP ensemble | 0.935 | +0.021 |
| + KDE PoE combination | 0.970 | +0.035 |
| + Retrieval features | 0.978 | +0.008 |
| + Irish translation | 0.985 | +0.007 |
| + Cleanlab noise-aware weighting | 0.988 | +0.003 |
| + Per-language KDE calibration | **0.990** | +0.002 |

The dominant factor is feature engineering: CES features alone account for 82% of the total QWK improvement from heuristic baseline to final system.

---

## Error Analysis

At QWK 0.990, the system makes 59 errors (1.42% error rate). Analysis of these errors reveals:

- **44% (26/59)** are on suspected label noise (cleanlab quality score < 0.4)
- **37% (22/59)** are on Irish-language items (7.3% of data but 37% of errors)
- **66% (39/59)** are boundary confusion (NC/PC or FC/PC)
- **30% (18/59)** are on short answers (< 100 characters)
- **5 high-error domains** account for 44% of all errors

See [docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md) for detailed analysis.

---

## Project Structure

```
sense-clef2026/
├── README.md
├── LICENSE                  # Apache 2.0
├── CITATION.cff
├── pyproject.toml
├── Dockerfile
├── .gitignore
├── sense/                   # Core package
│   ├── __init__.py
│   ├── models.py            # CORALHead, DeBERTaRubricScorer, datasets
│   ├── features.py          # V3 feature extraction (SBERT + NLI)
│   ├── ces.py               # CES subscore computation
│   ├── ensemble.py          # KDE PoE combination, retrieval scoring
│   ├── translate.py         # Irish detection + translation
│   └── io.py                # I/O, track detection, model loading
├── predict.py               # TIRA inference entrypoint
├── train.py                 # Full training pipeline
├── evaluate.py              # Evaluation against ground truth
├── docs/
│   ├── EXPERIMENT_LOG.md    # Full development timeline
│   └── ERROR_ANALYSIS.md    # Detailed error analysis at QWK=0.990
└── figures/
```

---

## Citation

```bibtex
@inproceedings{condrey2026sense,
  title     = {{SENSE}: Sensemaking Evaluation via {NLI} and Semantic Entailment},
  author    = {Condrey, David},
  booktitle = {Working Notes of CLEF 2026 -- Conference and Labs of the Evaluation Forum},
  year      = {2026}
}
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
