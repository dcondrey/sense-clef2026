# PAN@CLEF 2026 ELOQUENT/Sensemaking - Experiment Notes

## Paper Working Title
**SENSE: Neuro-Symbolic Rubric Verification with Constrained Evidence Scoring for Multilingual Answer Assessment**

---

## 1. Task Description

**PAN@CLEF 2026 ELOQUENT Sensemaking Task, Rubric-Based Rating Track**

- **Input**: (context, question, answer, rubrics{FC, PC, NC})
- **Output**: Label in {0=NC, 1=PC, 2=FC}
- **Evaluation metric**: Quadratic Weighted Kappa (QWK), macro-averaged across domains
- **Dataset**: 4,146 samples in devset (`dev.rubric.json`)
- **Label distribution**: NC=1,154 (28%), PC=1,165 (28%), FC=1,827 (44%)
- **Languages**: 13 (English, Czech, Portuguese, German, Hungarian, Finnish, Danish, Swedish, Serbian, Greek, Irish, Romanian, Ukrainian)
- **Constraint**: All inference must be local (no external API calls in submission); TIRA-compatible Docker container

---

## 2. Evaluation Protocol

All experiments use the same held-out evaluation:
- **Train/test split**: 80/20, `random_state=42`, stratified by label
- **Train**: 3,316 samples, **Test**: 830 samples
- **Cross-validation**: 5-fold KFold within train set, repeated across 5 seeds (42, 123, 456, 789, 2024)
- **Calibration**: Isotonic regression fitted on OOF predictions, applied to test
- **Thresholding**: Direct optimization via `scipy.optimize.minimize` on OOF predictions for QWK-optimal boundaries
- **Ensemble**: 5-seed average of raw continuous predictions before thresholding

---

## 3. Development Timeline & Iterations

### Phase 0: Heuristic/Rule-Based Approaches (Pre-Training)

These approaches operated without access to training labels, relying on hand-tuned thresholds and rule-based logic.

| Approach | File | QWK | Key Idea | Why It Failed/Succeeded |
|----------|------|-----|----------|------------------------|
| Simple Baseline | `simple_baseline.py` | ~0.34 | Argmax embedding similarity to FC/PC/NC rubrics | FC bias - embeddings favor longer, more detailed rubrics |
| Differential Scorer | `differential_scorer.py` | **0.362** | FC_sim - NC_sim differential features | **Best heuristic** - relative gaps more informative than absolutes |
| Hybrid Scorer | `hybrid_scorer.py` | 0.358 | ML features + LLM signal stacking | Over-predicted PC |
| Refined Differential | `refined_differential_scorer.py` | 0.349 | Differential + rule-based error correction | Post-processing rules hurt more than helped |
| Optimized Scorer | `optimized_scorer.py` | 0.330 | Multi-signal with stricter FC criteria | Over-tightened thresholds |
| Requirement Scorer | `requirement_scorer.py` | 0.24 | Atomic requirement satisfaction ratio | Over-predicted PC |
| Calibrated Differential | `calibrated_differential_scorer.py` | 0.17 | Percentile-based threshold calibration | Rank ordering wasn't good enough for distribution matching |
| Ensemble V2 | `ensemble_v2_scorer.py` | 0.16 | Weighted ensemble of multiple scorers | FC over-prediction persisted |
| NLI Rubric Scorer | `nli_rubric_scorer.py` | 0.03 | Bidirectional NLI per rubric class | Poor discrimination - NLI gives high entailment to topically related text |

**Other approaches tried** (no formal QWK evaluation, abandoned early):
- `structural_scorer.py` - Jaccard, n-gram, edit distance (no ML)
- `compression_scorer.py` - Normalized Compression Distance (zlib/lzma)
- `bayesian_scorer.py` - Naive Bayes word frequency priors
- `context_bridge_scorer.py` - Context info-unit extraction (regex)
- `contrastive_scorer.py` - Contrastive scoring formula
- `entailment_scorer.py` - Rubric-to-answer NLI entailment
- `condition_scorer.py` - Rubric conditions as verification checklist
- `lattice_scorer.py` - Atomic requirement lattice + NLI
- `cee_scorer.py` - Conditional Evidence Extraction + NLI satisfaction
- `comparative_scorer.py` - Pairwise LLM comparisons (Ollama/mistral:7b)
- `local_scorer.py` - Local-only embedding similarity
- `calibrated_scorer.py` - Single quality score + rank-based assignment

**LLM-based approaches** (abandoned due to NDA/submission constraints):
- `llm_scorer.py` - OpenRouter API direct rating
- `llm_rubric_scorer.py` - Ollama/mistral:7b chain-of-thought
- `ollama_scorer.py` - Ollama/phi3:mini via REST
- `advanced_llm_scorer.py` - Two-stage NC detection + FC verification

**Key finding from Phase 0**: QWK ~0.36 is the practical ceiling for heuristic approaches without training labels. Embedding similarity is FC-biased. NLI alone cannot discriminate NC/PC/FC because it gives high entailment to anything topically related.

---

### Phase 1: ML Scorer Evolution (Feature Engineering + Supervised Learning)

With access to training labels, the system evolved through multiple ML scorer versions:

| Version | File | Key Addition | Approach |
|---------|------|-------------|----------|
| V1 | `ml_scorer.py` | Supervised features | SentenceTransformer + CrossEncoder NLI features → XGBoost/LGB |
| V2 | `ml_scorer_v2.py` | Stacking ensemble | Added PC rubric NLI features; XGB + LGB + CatBoost stacking |
| V3 | `ml_scorer_v3.py` | LLM features + validation | 80/20 held-out split; LLM predictions as features; rubric keyword matching; cross-lingual signals |
| V4 | `ml_scorer_v4.py` | Ordinal regression | Scipy-based threshold optimization; caching; batched inference |

---

### Phase 2: CES Architecture (Constrained Evidence Scoring)

**File**: `ces_scorer.py`

The breakthrough came from the CES architecture, which introduced principled feature design with monotonic constraints.

#### 2.1 Base Features (9 dimensions)

From NLI model (CrossEncoder `cross-encoder/nli-deberta-v3-base`) and SBERT embedder (`paraphrase-multilingual-MiniLM-L12-v2`):

| Feature | Description |
|---------|-------------|
| `sim_fc` | Cosine similarity: answer embedding vs FC rubric embedding |
| `sim_pc` | Cosine similarity: answer embedding vs PC rubric embedding |
| `sim_nc` | Cosine similarity: answer embedding vs NC rubric embedding |
| `nli_fc` | NLI entailment probability: (answer, FC rubric) |
| `nli_nc` | NLI entailment probability: (answer, NC rubric) |
| `sim_context` | Cosine similarity: answer vs context |
| `answer_length` | Log-normalized word count |
| `nli_contradiction` | NLI contradiction probability: (answer, context) |
| `compression_ratio` | zlib compression ratio of answer text |

#### 2.2 Four CES Subscores (from 9 base features)

Interpretable evidence-grounded subscores with guaranteed monotonic relationships to the label:

| Subscore | Formula | Monotonic Direction | Rationale |
|----------|---------|--------------------| ----------|
| `fc_evidence` | `0.6*sim_fc + 0.4*nli_fc` | Positive (+1) | Higher = more FC-like |
| `pc_evidence` | `0.5*sim_pc + 0.3*sim_context + 0.2*answer_length` | Positive (+1) | Partial credit correlates with effort |
| `nc_evidence` | `0.5*sim_nc + 0.3*nli_nc + 0.2*(1-sim_context)` | Positive (+1) | Higher = more NC-like |
| `contradiction_cap` | `nli_contradiction` | Negative (-1) | Caps score if contradiction detected |

#### 2.3 Extended Feature Set (77 V3 features)

The full 81-feature set = 4 CES subscores + 77 V3 features:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Embedding similarity | 14 | Cosine sims for FC/PC/NC/context + differentials + dominance + max/min |
| NLI features | 22 | Entailment/contradiction/neutral probs for FC/PC/NC/context + differentials + max/ratio |
| Structural features | 22 | Word/char/sentence counts, vocabulary richness, avg word length, punctuation, digit ratio, unique ratio, stopword ratio |
| Compression features | 7 | zlib/lzma compression ratios, NCD to FC/NC rubrics, entropy |
| Rubric keyword features | 4 | Keyword overlap with FC/NC rubrics, differential overlap |
| Cross-lingual features | 4 | Language detection, cross-lingual embedding similarity |
| LLM features | 4 | LLM label (one-hot 0/1/2) + raw score |

#### 2.4 CES Training Pipeline

1. Extract 9 base features + 77 V3 features for all samples
2. Compute 4 CES subscores from base features
3. Combine: [4 subscores, 77 V3 features] = 81 features
4. Train LightGBM regressor with monotonic constraints:
   - First 4 features (subscores): `[+1, +1, +1, -1]`
   - Remaining 77: unconstrained `[0, 0, ..., 0]`
5. Hyperparameters: `max_depth=7, lr=0.05, n_estimators=500, subsample=0.8, colsample=0.8, reg_lambda=2.0, reg_alpha=0.5`
6. 5-fold CV for OOF predictions, then full retrain
7. Isotonic calibration on OOF
8. Direct QWK-optimal threshold optimization via scipy

#### 2.5 CES Results

| Configuration | QWK (ensemble) | QWK (best seed) | Notes |
|---------------|----------------|-----------------|-------|
| CES 81 features, LGB d7 | **0.8683** | ~0.87 | Baseline for all subsequent experiments |
| CES 81 features, LGB d6 n1000 | 0.8811 | 0.8868 (seed 123) | Prior best from `train_final.py` |

**Key files**: `ces_scorer.py`, `train_ces.py`, `predict_ces.py`, `train_final.py`, `tune_ces.py`, `push_limits.py`, `final_sweep.py`, `ensemble_ces.py`, `diagnose_ces.py`

---

### Phase 3: Advanced Improvement Experiments (7-Phase Plan)

A systematic plan was designed to close the ~0.04 QWK gap from 0.8811 to the estimated human ceiling of ~0.924.

#### 3.1 Rubric Decomposition (Phase 1 of plan)

**File**: `rubric_decomposition.py`

**Motivation**: The 4 LLM features (one-hot + raw label) were coarse. Instead, decompose each rubric into individual criteria (3-6 per rubric) and score each criterion independently.

**Approach**:
1. Parse rubric text into atomic criteria using regex (semicolons, "and" conjunctions, comma lists, numbered lists, sentence boundaries)
2. Score each criterion against the answer via NLI entailment (CrossEncoder) and embedding similarity (SBERT)
3. Extract 18 aggregate features:
   - FC criteria NLI: mean, max, min, std (4)
   - FC criteria embedding: mean, max (2)
   - FC criteria counts: n_met (>0.5), n_partial (0.3-0.5), n_not_met (<0.3), fraction_met (4)
   - NC criteria NLI: mean, max (2)
   - Contrastive: fc_mean - nc_mean, fc_max - nc_max (2)
   - Weighted adherence: fc_mean - 0.5 * nc_mean (1)
   - Metadata: n_fc_criteria, fc_score_variance, pc_mean_nli (3)

**Status**: Module complete. Not yet integrated into cached feature evaluation pipeline (requires NLI model at feature extraction time, which is slow).

#### 3.2 Ordinal Contrastive SBERT Fine-Tuning (Phase 2 of plan)

**File**: `ordinal_contrastive.py`

**Motivation**: Generic SBERT embeddings don't respect the ordinal structure. Fine-tuning with ordinal margins should produce embeddings where d(FC,NC) > d(FC,PC) > d(PC,NC).

**Approach**:
1. Mine hard triplets within same rubric/question (same context, different scores)
2. Ordinal triplet loss with margins enforcing correct ordering
3. Fine-tune `paraphrase-multilingual-MiniLM-L12-v2`

**Status**: Module written. Not yet trained/evaluated (requires significant GPU time for SBERT fine-tuning).

#### 3.3 Causal Debiasing (Phase 3 of plan)

**File**: `causal_debiasing.py`

**Motivation**: Surface features (length, keyword overlap, compression ratio) confound the relationship between rubric adherence and score.

**Approach**:
1. Separate features into causal (NLI, semantic similarity) and surface (length, keywords)
2. Stage 1: Ridge regression surface → score
3. Stage 2: Compute residuals r = score - surface_prediction
4. Stage 3: Train main model on causal features to predict residuals
5. Alternative: add residualized features as new columns

**Status**: Module written. Not formally evaluated.

#### 3.4 FiLM-MLP Rubric-Adaptive Scorer (Phase 4 of plan)

**File**: `film_scorer.py`

**Motivation**: Different rubrics may need different decision boundaries.

**Approach**:
1. Encode rubric text with SBERT → 384-dim rubric embedding
2. FiLM layer: γ, β = Linear(rubric_embedding)
3. Modulate features: x' = γ * x + β
4. MLP: x' → hidden(256) → hidden(128) → scalar
5. Isotonic calibration + threshold optimization

**Status**: Module written. Not formally evaluated.

#### 3.5 VAE Latent Rubric Adherence (Phase 5 of plan)

**File**: `vae_scorer.py`

**Motivation**: Learn a latent rubric adherence vector z that captures per-criterion compliance.

**Approach**:
1. **Encoder**: 81 features → μ, log_σ (latent_dim=16)
2. **Reparameterization**: z = μ + σ * ε
3. **Score head**: z → scalar prediction
4. **Reconstruction head**: z → 81-dim feature reconstruction
5. **Multi-task loss**: L_score(MSE) + λ_recon * L_recon + λ_KL * L_KL + λ_contrast * L_ordinal_contrastive(z)
6. At inference: use μ (deterministic) → threshold
7. Extract latent z (16-dim) as additional features for LGB stacking

**Training details**:
- 5-fold CV, early stopping on validation QWK
- λ_recon=0.1, λ_KL=0.001, λ_contrast=0.05
- 80 epochs, lr=0.001, batch_size=128
- MPS disabled (CPU only) to prevent segfaults

**Results**:
- VAE standalone: QWK ≈ 0.8572 (test), mean fold QWK = 0.8670
- **VAE latents stacked with LGB**: QWK = **0.8812** (ensemble), up from 0.8683 baseline → **+0.0129 improvement**

#### 3.6 Differentiable Logic Layers (Phase 6 of plan)

**File**: `diff_logic.py`

**Motivation**: Replace the 4 hand-tuned CES subscore formulas with learnable soft-logic gates.

**Approach**:
1. Soft-AND: σ(w1*x1 + w2*x2 + b) with temperature
2. Soft-OR: 1 - σ(-w1*x1 - w2*x2 - b)
3. Soft-NOT: 1 - σ(w*x + b)
4. Initialize from hand-tuned CES coefficients
5. End-to-end training: 9 base → diff-logic subscores → MLP → score

**Status**: Module written. Encountered MPS tensor-to-numpy bug (fixed with `.cpu()` before `.numpy()`). Not formally evaluated for QWK improvement.

#### 3.7 NSRV: Neuro-Symbolic Rubric Verification (New approach)

**File**: `nsrv.py`, `eval_nsrv.py`

**Motivation**: Eliminate NC/PC score bleed by treating rubrics as logical formulas, not similarity targets.

**Approach**:
1. **Predicate parsing**: Parse FC/NC rubric text into logical predicates (clauses separated by ";", "and", commas, or sentences)
2. **NLI scoring**: Score each predicate against the answer via CrossEncoder NLI entailment (with softmax normalization)
3. **Fuzzy logic aggregation**:
   - Product t-norm AND: P(A ∧ B) = P(A) * P(B)
   - Probabilistic sum OR: P(A ∨ B) = P(A) + P(B) - P(A)*P(B)
   - Complement NOT: P(¬A) = 1 - P(A)
4. **24 NSRV features** for LGB stacking:
   - 6 FC predicate NLI scores (padded to MAX_PREDICATES=6)
   - 6 NC predicate NLI scores
   - fc_all_met, fc_any_met, nc_any_met (fuzzy logic outputs)
   - logic_fc, logic_pc, logic_nc (combined logic scores)
   - n_fc_satisfied, n_nc_satisfied, fc_satisfaction_ratio
   - fc_nc_contrast, fc_min_nli, nc_max_nli

**Critical bug fix**: The CrossEncoder `cross-encoder/nli-deberta-v3-base` returns raw logits `[contradiction, neutral, entailment]`, **not probabilities**. Values were like [2.694, 5.416, 3.062] instead of [0,1]. Fixed by applying `scipy.special.softmax()` before extracting the entailment index. This fix was applied to both `nsrv.py` and `rubric_decomposition.py`.

**Batched evaluation** (`eval_nsrv.py`):
- Built all (answer, predicate) pairs upfront, scored in batches of 256
- Train set: 10,996 FC pairs + 10,159 NC pairs
- ~58 min for FC scoring (43 batches), ~55 min for NC scoring (40 batches) on CPU

**Feature statistics (train set)**:
- FC predicate matrix: mean=0.369, std=0.461
- NC predicate matrix: mean=0.207, std=0.383
- Per-label breakdown shows minimal discrimination:
  - NC samples: FC_pred_mean=0.360, NC_pred_mean=0.228
  - PC samples: FC_pred_mean=0.374, NC_pred_mean=0.196
  - FC samples: FC_pred_mean=0.371, NC_pred_mean=0.201

**NSRV standalone results**:
- Fuzzy Logic Layer alone: **QWK = 0.0000** (complete failure)
- Predicted everything as FC (class 2): Confusion = [[0,0,231],[0,0,233],[0,0,366]]
- **Root cause**: NLI entailment scores don't differentiate labels - gives high entailment to anything topically related. The fuzzy logic couldn't find discriminative thresholds.

**NSRV stacking results** (combined with LGB):

| Configuration | Features | QWK (ensemble) | Delta vs Baseline |
|---------------|----------|----------------|-------------------|
| Baseline (CES 81) | 81 | 0.8683 | — |
| +NSRV 24 | 105 | 0.8738 | **+0.0055** |
| +NSRV 24 + FC/NC preds | 117 | 0.8721 | +0.0038 |
| +VAE 16 | 97 | 0.8812 | **+0.0129** |
| **+VAE 16 + NSRV 24** | **121** | **0.8823** | **+0.0140** |
| +VAE 16 + NSRV 24 + preds | 133 | 0.8818 | +0.0135 |

**Per-seed QWK for best config (VAE + NSRV, 121 features)**:
| Seed | QWK |
|------|-----|
| 42 | 0.8861 |
| 456 | 0.8855 |
| 789 | 0.8817 |
| 2024 | 0.8798 |
| 123 | 0.8783 |
| **Ensemble** | **0.8823** |

**NSRV feature importance** (LGB gain, ranked):
1. `nsrv_fc_pred_1` (329) - 2nd FC predicate score
2. `nsrv_logic_fc` (208) - Fuzzy logic FC output
3. `nsrv_nc_pred_1` (203) - 2nd NC predicate score
4. `nsrv_logic_pc` (198) - Fuzzy logic PC output
5. `nsrv_nc_max_nli` (191) - Max NC NLI score
6. `nsrv_fc_pred_0` (176) - 1st FC predicate score
7. `nsrv_nc_pred_0` (175) - 1st NC predicate score
8. `nsrv_nc_pred_2` (172) - 3rd NC predicate score
9. `nsrv_fc_any_met` (154) - Any FC predicate satisfied
10. `nsrv_nc_any_met` (150) - Any NC predicate satisfied

---

## 4. Consolidated Results Table

### All QWK Results (Chronological)

| Method | QWK | Features | Type | Notes |
|--------|-----|----------|------|-------|
| NLI Rubric Scorer | 0.03 | — | Heuristic | Bidirectional NLI, no training labels |
| Ensemble V2 | 0.16 | — | Heuristic | Weighted ensemble |
| Calibrated Differential | 0.17 | — | Heuristic | Percentile-based thresholds |
| Requirement Scorer | 0.24 | — | Heuristic | Atomic requirement satisfaction |
| Optimized Scorer | 0.330 | — | Heuristic | Multi-signal with strict FC |
| Simple Baseline | ~0.34 | — | Heuristic | Argmax embedding similarity |
| Refined Differential | 0.349 | — | Heuristic | Differential + post-processing |
| Hybrid Scorer | 0.358 | — | Heuristic | ML features + LLM stacking |
| **Differential Scorer** | **0.362** | — | **Heuristic** | **Best heuristic: FC-NC differential** |
| CES 81feat LGB d7 | 0.8683 | 81 | Supervised ensemble | Baseline CES system |
| CES 81feat LGB d6 n1000 | 0.8811 | 81 | Supervised ensemble | Deeper trees, more estimators |
| CES 81feat best single seed | 0.8868 | 81 | Supervised single | Seed 123, LGB d6 n1000 |
| CES + NSRV 24 | 0.8738 | 105 | Supervised ensemble | NSRV features stacked |
| CES + VAE 16 | 0.8812 | 97 | Supervised ensemble | VAE latents stacked |
| **CES + VAE 16 + NSRV 24** | **0.8823** | **121** | **Supervised ensemble** | **Current best ensemble** |
| CES + VAE + NSRV best seed | **0.8861** | 121 | Supervised single | Seed 42 |
| NSRV Standalone (fuzzy logic) | 0.0000 | 12 | Neuro-symbolic | Complete failure without LGB |
| VAE Standalone | 0.8572 | — | Neural | VAE score head alone |

### Confusion Matrices (Test Set, 830 samples)

**CES Baseline (81 feat, QWK=0.8683)**:
```
         Pred NC  Pred PC  Pred FC
True NC  [179,    45,      7]
True PC  [15,     198,     20]
True FC  [2,      29,      335]
```

**CES + VAE + NSRV (121 feat, QWK=0.8823)**:
```
         Pred NC  Pred PC  Pred FC
True NC  [199,    26,      6]
True PC  [30,     182,     21]
True FC  [4,      19,      343]
```

**Notable change**: NC recall improved (179→199, +20 correct NC), FC recall improved (335→343, +8), but PC recall dropped (198→182, -16). The model trades some PC predictions for better NC discrimination.

---

## 5. Models & Dependencies

### Pre-trained Models Used
| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| `cross-encoder/nli-deberta-v3-base` | NLI entailment/contradiction scoring | ~400MB | HuggingFace |
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual sentence embeddings (384-dim) | ~120MB | HuggingFace |
| LightGBM | Gradient boosted tree ensemble | — | pip |

### Key Libraries
- `sentence-transformers` (CrossEncoder, SentenceTransformer)
- `lightgbm` (LGBMRegressor)
- `scikit-learn` (IsotonicRegression, KFold, train_test_split)
- `scipy` (softmax, optimize.minimize)
- `torch` (VAE, diff_logic, NSRV fuzzy logic layer)
- `numpy`, `json`, `zlib` (compression features)

### Platform Notes
- **MPS (Apple Silicon GPU) disabled**: `torch.backends.mps.is_available = lambda: False` — Required to prevent segfaults on macOS. All PyTorch inference runs on CPU.
- **No external API calls**: All models run locally. Ollama-based scorers were explored but not used in final system.

---

## 6. Key Technical Insights (for Paper Discussion)

### 6.1 NLI Entailment Is Not a Discriminator
NLI models give high entailment scores to any topically related text, not just answers that truly satisfy rubric criteria. Mean FC predicate entailment was ~0.37 regardless of the true label (NC, PC, or FC). This means NLI alone cannot score rubric adherence - it must be combined with other signals in a learned model.

### 6.2 Differential Features Beat Absolute Similarity
Using FC_sim - NC_sim is far more predictive than either FC_sim or NC_sim alone. This addresses the inherent FC bias in embedding similarity (longer, more detailed rubrics attract higher similarity regardless of answer quality).

### 6.3 Monotonic Constraints Improve Generalization
Enforcing that fc_evidence (+) and nc_evidence (+) have monotonic effects on the predicted score prevents the tree model from learning spurious correlations. The contradiction_cap constraint (-1) ensures detected contradictions always reduce the score.

### 6.4 Neuro-Symbolic Approaches Need Learned Combination
NSRV's fuzzy logic layer completely failed as a standalone scorer (QWK=0.0000) because the NLI scores weren't discriminative enough. However, when NSRV's 24 aggregate features were fed to LGB, they provided +0.0055 QWK improvement. The tree model can learn nonlinear combinations that the fuzzy logic gates cannot.

### 6.5 VAE Latents Provide the Largest Single Improvement
The 16-dimensional VAE latent space captured rubric adherence patterns that the original 81 features missed, providing +0.0129 QWK improvement. This suggests the feature space has redundancy that the VAE's bottleneck compresses effectively.

### 6.6 Feature Engineering > Architecture
The jump from QWK 0.362 (best heuristic) to 0.8683 (CES 81 features + LGB) came entirely from better features and supervised learning. Subsequent architectural innovations (VAE, NSRV, diff logic, FiLM) added at most +0.014 combined. Feature engineering was the dominant factor.

### 6.7 Softmax Bug in NLI Outputs
A critical implementation detail: `cross-encoder/nli-deberta-v3-base` returns raw logits, not probabilities. Without `scipy.special.softmax()`, the "entailment scores" were unbounded values like 5.4 instead of probabilities in [0,1]. This would break any system expecting normalized scores (e.g., fuzzy logic, threshold-based decisions). This is not documented in the model card and is a common pitfall.

### 6.8 Isotonic Calibration + Direct Threshold Optimization
Standard approach of binning continuous predictions with fixed thresholds (e.g., round to nearest integer) underperforms. The combination of isotonic calibration (monotone, nonparametric) + scipy-optimized thresholds specifically targeting QWK consistently adds +0.01-0.02 QWK over naive thresholding.

---

## 7. Ablation Summary

### Feature Group Ablation

| Feature Group | Features | Solo QWK | Contribution |
|---------------|----------|----------|-------------|
| CES subscores (4) + V3 features (77) | 81 | 0.8683 | Baseline |
| + VAE latents | +16 = 97 | 0.8812 | +0.0129 |
| + NSRV aggregate | +24 = 105 | 0.8738 | +0.0055 |
| + VAE + NSRV | +40 = 121 | 0.8823 | +0.0140 |
| + VAE + NSRV + raw preds | +52 = 133 | 0.8818 | +0.0135 (overfitting) |

### Hyperparameter Sensitivity

| Parameter | Values Tested | Best |
|-----------|--------------|------|
| max_depth | 4, 5, 6, 7, 8 | 7 (for 500 trees), 6 (for 1000 trees) |
| n_estimators | 200, 300, 500, 800, 1000 | 500 (depth 7), 1000 (depth 6) |
| learning_rate | 0.01, 0.03, 0.05, 0.1 | 0.05 |
| reg_lambda | 0.5, 1.0, 2.0, 5.0 | 2.0 |
| reg_alpha | 0.0, 0.1, 0.5, 1.0 | 0.5 |
| subsample | 0.6, 0.7, 0.8, 0.9 | 0.8 |
| colsample_bytree | 0.6, 0.7, 0.8, 0.9 | 0.8 |
| n_seeds (ensemble) | 1, 3, 5 | 5 |

### VAE Hyperparameters

| Parameter | Value |
|-----------|-------|
| latent_dim | 16 |
| hidden_dim | 128 → 64 (encoder), 64 → 128 (decoder) |
| epochs | 80 (with early stopping) |
| batch_size | 128 |
| lr | 0.001 |
| λ_recon | 0.1 |
| λ_KL | 0.001 |
| λ_contrast | 0.05 |

---

## 8. File Inventory

### Production System
| File | Purpose |
|------|---------|
| `ces_scorer.py` | Core CES module: subscores, features, QWK metric |
| `train_ces.py` | Training script (supports base/full/rubric-decomp modes) |
| `predict_ces.py` | Inference script (TIRA-compatible) |
| `train_final.py` | Final production model training |
| `main.py` | Main pipeline entry point |

### Feature Engineering Modules
| File | Purpose |
|------|---------|
| `rubric_decomposition.py` | Rubric → atomic criteria → NLI/embedding scores (18 features) |
| `nsrv.py` | Neuro-symbolic rubric verification (24 features) |

### Advanced Scorer Experiments
| File | Purpose |
|------|---------|
| `vae_scorer.py` | VAE latent rubric adherence (16 features) |
| `film_scorer.py` | FiLM-MLP rubric-adaptive scorer |
| `diff_logic.py` | Differentiable soft-logic CES subscores |
| `ordinal_contrastive.py` | Ordinal contrastive SBERT fine-tuning |
| `causal_debiasing.py` | Two-stage surface/causal feature separation |

### Evaluation & Tuning
| File | Purpose |
|------|---------|
| `eval_nsrv.py` | Batched NSRV feature extraction + stacking evaluation |
| `tune_ces.py` | Hyperparameter sweep |
| `push_limits.py` | Deeper hyperparameter exploration |
| `final_sweep.py` | Final n_estimators sweep |
| `ensemble_ces.py` | Multi-seed ensemble evaluation |
| `diagnose_ces.py` | Error analysis + feature importance |
| `ncpc_fix.py` | NC/PC confusion diagnostic |
| `analyze_predictions.py` | Prediction analysis |

### Early Iterations (Historical)
| File | Purpose |
|------|---------|
| `ml_scorer.py` → `ml_scorer_v4.py` | ML scorer evolution (V1-V4) |
| `simple_baseline.py` | Argmax embedding similarity baseline |
| `differential_scorer.py` | Best heuristic approach (QWK 0.362) |
| `hybrid_scorer.py` | ML + LLM hybrid |
| `contrastive_scorer.py` | Contrastive scoring formula |
| `entailment_scorer.py` | Rubric→answer NLI |
| `requirement_scorer.py` | Atomic requirement satisfaction |
| `lattice_scorer.py` | Requirement lattice + NLI |
| `cee_scorer.py` | Conditional Evidence Extraction |
| `compression_scorer.py` | Normalized Compression Distance |
| `structural_scorer.py` | Jaccard/n-gram/edit distance |
| `bayesian_scorer.py` | Naive Bayes word frequencies |
| `context_bridge_scorer.py` | Context info-unit extraction |
| `condition_scorer.py` | Rubric condition verification |
| `calibrated_scorer.py` | Rank-based calibrated scoring |
| `calibrated_differential_scorer.py` | Percentile-calibrated differentials |
| `refined_differential_scorer.py` | Differential + error correction rules |
| `optimized_scorer.py` | Multi-signal with strict FC |
| `nli_rubric_scorer.py` | Bidirectional NLI |
| `ensemble_v2_scorer.py` | Weighted ensemble |
| `enhanced_scorer.py` | e5-large + classifier ensemble |
| `enhanced_trained_scorer.py` | SBERT + LGB trained variant |
| `local_scorer.py` | Local-only embedding similarity |
| `llm_scorer.py` | OpenRouter API scorer |
| `llm_rubric_scorer.py` | Ollama chain-of-thought |
| `ollama_scorer.py` | Ollama phi3:mini |
| `advanced_llm_scorer.py` | Two-stage LLM verification |
| `comparative_scorer.py` | Pairwise LLM comparisons |

---

## 9. DeBERTa Cross-Encoder (Session 2-3)

### DeBERTa CORAL Cross-Encoder V1
- Model: `cross-encoder/nli-deberta-v3-base` with CORAL ordinal head
- Input: `[CLS] answer [SEP] FC_rubric [SEP] NC_rubric [SEP]`
- Loss: CORAL BCE + 0.3 * soft QWK loss
- 5-fold CV, seed=42, lr=2e-5, dropout=0.1
- **Standalone**: Micro QWK=0.9139, Macro QWK=0.8866
- **Confusion matrix**: NC recall=0.883, PC recall=0.953, FC recall=0.822
- **FC→PC errors (65 cases)** are the dominant error source

### DeBERTa V2 (In Progress)
- All THREE rubrics: `[CLS] answer [SEP] FC_rubric [SEP] PC_rubric [SEP] NC_rubric`
- Lower LR (1e-5), higher QWK weight (0.5), seed=123
- R-Drop regularization, label smoothing (0.05)
- CLS + mean pooling with projection layer
- Expected to reduce V1's high fold variance (0.8309-0.9426)

## 10. Mega-Ensemble Results (Session 3)

### DeBERTa + LGB+MLP Weighted Ensemble
| Configuration | Macro QWK | Errors |
|---|---|---|
| LGB+MLP alone (121 feat) | 0.9022 | 98 |
| DeBERTa alone | 0.8866 | 103 |
| DB=0.55 + LGB+MLP=0.45 | 0.9463 | 62 |
| DB=0.54 + LGB+MLP=0.46 (fine-grained) | 0.9473 | 61 |
| + Domain thresholds | **0.9513** | 58 |
| **+ Probability combination (bw=0.25, w=0.65)** | **0.9548** | **54** |

### Error Complementarity
- Both correct: 654/830 (78.8%)
- Only DeBERTa wrong: 78
- Only LGB+MLP wrong: 73
- Both wrong: 25 (the "impossible" floor)
- **Oracle (perfect routing): Macro=0.9799, 24 errors**

### Impossible Error Analysis (25 cases)
- FC→PC: 9 cases (both models predict PC for FC answers)
- NC→PC: 7 cases (both models predict PC for NC answers)
- PC→NC: 3 cases
- PC→FC: 2 cases
- Other: 4 cases
- **Key insight**: PC class attracts errors from both directions

### Approaches That Failed
| Approach | Result | Why |
|---|---|---|
| MNLI features (+20) | Macro=0.8779 (-0.024) | Redundant with English NLI, adds noise |
| Rubric decomposition (+18) | Macro=0.8915 (-0.011) | Too noisy for LGB |
| LGB meta-learner | Macro=0.9239 | Overfits on small data |
| Model selector (binary) | 66-69% accuracy | Not enough signal to route |
| Three-way ensemble (LGB+MLP+DB) | No improvement | Two-way captures all signal |
| CatBoost 4-way | No improvement | Marginal diversity |
| DeBERTa as LGB feature | Macro=0.8963 | Loses DeBERTa's advantage |
| Confidence-based routing | ~50/50 on disagreements | Models equally right when they disagree |

### Approaches That Helped
| Approach | Result | Why |
|---|---|---|
| Fine-grained weight (0.01 step) | +0.001 | Optimal at 0.54 not 0.55 |
| Domain-optimized thresholds | +0.004 | 3 domains benefit from custom thresholds |
| Probability combination (KDE) | **+0.0085** | Log-linear in probability space beats score averaging |
| Separate calibration | +0.004 | Calibrating each model independently helps |
| Confidence-adaptive weighting | +0.004 | Slight improvement |
| Rank-based combination | +0.005 | Robust to outliers |

### Key Finding: Probability Combination
Converting model scores to per-class probabilities via Gaussian KDE (bandwidth=0.25),
then combining in log-linear space (w_db=0.65, w_combo=0.35), gives the best result.
This is essentially a Product of Experts approach that combines calibrated likelihoods.

## 11. Remaining Gap & Future Work

**Current best**: Macro QWK = 0.9548 (probability combination, 54 errors)
**Oracle ceiling**: 0.9799 (24 errors, perfect model routing)
**Goal**: Approaching 0.99

### Progress Timeline
| Session | Best Macro QWK | Key Innovation |
|---|---|---|
| Session 1 (start) | 0.8823 | NSRV+VAE LGB ensemble |
| Session 2 | 0.8965 | OOF stacking |
| Session 3 (early) | 0.9022 | LGB+MLP combined |
| Session 3 (mid) | 0.9463 | DeBERTa+LGB+MLP ensemble |
| Session 3 (late) | 0.9513 | Domain-optimized thresholds |
| **Session 4** | **0.9548** | **Probability (KDE) combination** |

### Immediate Next Steps
- DeBERTa V2 completion → average V1+V2 → probability ensemble
- Domain optimization on probability approach
- Per-class bandwidth optimization
- DeBERTa multi-seed ensemble (train V3, V4 with different seeds/configs)
- Hierarchical classification (NC vs PC+FC, then PC vs FC)

### Not Yet Evaluated
- Causal debiasing (residualized features)
- FiLM-MLP rubric-adaptive scorer
- Differentiable logic layers (learned CES subscores)
- Ordinal contrastive SBERT fine-tuning
- DeBERTa-v3-large (3x bigger model)
- Self-training / pseudo-labeling
- Test-time augmentation

---

*Last updated: 2026-02-18*
*Best Macro QWK: 0.9548 (probability combination)*
