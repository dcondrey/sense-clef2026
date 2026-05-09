# Error Analysis Report: QWK=0.990 Classification System
## PAN@CLEF 2026 Sensemaking Task

**Date**: 2026-02-20
**System**: push_qwk_submission.json
**Performance**: QWK = 0.990, 59/4146 errors (1.42% error rate)

---

## Executive Summary

This report analyzes the remaining 59 errors from a high-performing QWK=0.990 classification system for PAN@CLEF 2026 sensemaking (rubric-based rating: NC=0, PC=1, FC=2).

**Key Findings**:
- **44% of errors (26/59)** are on suspected noisy labels per cleanlab analysis
- **37% of errors (22/59)** are on Irish (ga) language items, despite Irish being only 7.3% of dataset
- **66% of errors (39/59)** are boundary confusion errors (NC→PC or FC→PC)
- **5 high-error domains** account for 44% of all errors (26/59)
- **30% of errors (18/59)** occur on short answers (<100 characters)

---

## 1. Error Type Breakdown

### Overall Error Distribution (59 total errors)

| Error Type | Count | Percentage | Description |
|------------|-------|------------|-------------|
| **NC→PC** | 23 | 39.0% | Model over-predicts PC (false positive for partial correctness) |
| **FC→PC** | 16 | 27.1% | Model under-predicts FC (misses fully correct answers) |
| **PC→FC** | 12 | 20.3% | Model over-predicts FC (gives too much credit) |
| **PC→NC** | 8 | 13.6% | Model under-predicts PC (too harsh) |

**Boundary Confusion**: 39/59 errors (66.1%) are NC→PC or FC→PC, indicating the model struggles most with identifying the PC boundary.

### Answer Length Patterns by Error Type

| Error Type | Mean Length | Median Length | Distribution (<100 / 100-300 / 300+) |
|------------|-------------|---------------|--------------------------------------|
| **NC→PC** | 210.7 chars | 202 chars | 4 / 11 / 8 |
| **FC→PC** | 114.9 chars | 137.5 chars | 6 / 10 / 0 |
| **PC→FC** | 215.8 chars | 150.5 chars | 6 / 1 / 5 |
| **PC→NC** | 211.4 chars | 249 chars | 2 / 4 / 2 |

**Key Insight**: FC→PC errors tend to occur on **shorter answers** (mean 115 chars), while NC→PC errors occur on **longer answers** (mean 211 chars). This suggests the model may be over-relying on answer length as a feature.

---

## 2. Language-Specific Analysis

### Per-Language Error Rates

| Language | Total Items | Errors | Error Rate | % of Total Errors |
|----------|-------------|--------|------------|-------------------|
| **ga (Irish)** | 302 | 22 | **7.28%** | **37.3%** |
| uk (Ukrainian) | 320 | 6 | 1.88% | 10.2% |
| el (Greek) | 302 | 5 | 1.66% | 8.5% |
| en (English) | 344 | 5 | 1.45% | 8.5% |
| fi (Finnish) | 302 | 4 | 1.32% | 6.8% |
| pt (Portuguese) | 326 | 4 | 1.23% | 6.8% |
| sr (Serbian) | 332 | 4 | 1.20% | 6.8% |
| hu (Hungarian) | 320 | 3 | 0.94% | 5.1% |
| sv (Swedish) | 326 | 3 | 0.92% | 5.1% |
| de (German) | 300 | 2 | 0.67% | 3.4% |
| cs (Czech) | 332 | 1 | 0.30% | 1.7% |
| da (Danish) | 314 | 0 | 0.00% | 0.0% |
| ro (Romanian) | 326 | 0 | 0.00% | 0.0% |

**Critical Finding**: Irish (ga) has a **7.28% error rate**, **5.1x higher** than the overall 1.42% error rate.

### Irish (ga) Error Type Distribution

**22 total Irish errors (37.3% of all errors)**

| Error Type | Count | % of Irish Errors | % of All Errors of This Type |
|------------|-------|-------------------|------------------------------|
| **NC→PC** | 13 | 59.1% | 56.5% of all NC→PC errors |
| **PC→FC** | 4 | 18.2% | 33.3% of all PC→FC errors |
| **FC→PC** | 3 | 13.6% | 18.8% of all FC→PC errors |
| **PC→NC** | 2 | 9.1% | 25.0% of all PC→NC errors |

**Irish-Specific Pattern**: The model has a strong tendency to **over-predict PC for Irish NC answers** (13/22 = 59% of Irish errors are NC→PC).

### Language-Specific Error Type Patterns

| Error Type | Irish (22 errors) | Other Languages (37 errors) |
|------------|-------------------|-----------------------------|
| **NC→PC** | 13 (59.1%) | 10 (27.0%) |
| **FC→PC** | 3 (13.6%) | 13 (35.1%) |
| **PC→FC** | 4 (18.2%) | 8 (21.6%) |
| **PC→NC** | 2 (9.1%) | 6 (16.2%) |

**Insight**: Irish errors are dominated by NC→PC (over-prediction), while other languages have more balanced error distributions with FC→PC being most common.

---

## 3. Cleanlab Label Noise Analysis

### Noise Detection Results

- **Total suspicious items in training set**: 77 (quality score < 0.4)
- **Label quality scores**: min=0.009, mean=0.823, max=0.963

### Error Breakdown by Noise Status

| Category | Count | Percentage |
|----------|-------|------------|
| **Suspected label noise (train set)** | 26 | 44.1% |
| **Real model errors (train set)** | 11 | 18.6% |
| **Test set errors** | 22 | 37.3% |

**Critical Insight**: **44% of errors (26/59)** are on items suspected to be mislabeled according to cleanlab analysis. These may not represent genuine model failures.

### Suspected Noise Errors by Type

| Error Type | Count | Top Languages | Quality Score Range |
|------------|-------|---------------|---------------------|
| **NC→PC** | 11 | ga(5), pt(2), uk(1), sv(1), fi(1), hu(1) | 0.076 - 0.266 |
| **FC→PC** | 5 | ga(1), el(1), uk(1), fi(1), en(1) | 0.031 - 0.310 |
| **PC→FC** | 5 | el(2), ga(1), uk(1), fi(1) | 0.132 - 0.396 |
| **PC→NC** | 5 | sr(2), ga(2), en(1) | 0.048 - 0.400 |

### Real Model Errors (High Confidence Labels) by Type

| Error Type | Count | Languages | Quality Score Range |
|------------|-------|-----------|---------------------|
| **FC→PC** | 4 | de, pt, sv, sr | 0.460 - 0.528 |
| **PC→FC** | 4 | fi, sr, hu, de | 0.408 - 0.462 |
| **PC→NC** | 2 | cs, pt | 0.497 - 0.516 |
| **NC→PC** | 1 | el | 0.482 |

**Only 11 train errors** (18.6% of total) are on high-confidence labels (quality > 0.4), suggesting these are genuine model mistakes rather than label noise.

---

## 4. Domain-Specific Patterns

### High-Error Domains (3+ errors each)

#### Domain 1: "Ancestry influences prevalence..." (10 errors)
- **Error types**: PC→NC(3), PC→FC(1), FC→PC(3), NC→PC(3)
- **Languages**: sr(1), el(2), uk(1), ga(3), sv(1), en(1), hu(1)
- **Pattern**: Highly confused domain with all error types present
- **Priority**: **CRITICAL** - Consider domain-specific threshold calibration

#### Domain 2: "Study answered 'no' questions" (4 errors)
- **Error types**: PC→NC(2), NC→PC(2)
- **Languages**: sr(1), cs(1), ga(2)
- **Pattern**: Symmetric confusion between NC and PC
- **Example errors**:
  - ID 50398 (sr, PC→NC): "Pustiti da potone i raspadne se? Može li korištenje više proizvoda od algi smanjiti emisije CO2?"
  - ID 84610 (cs, PC→NC): "Nechat to potopit a rozložit? Mohlo by použití více produktů z mořských řas snížit emise CO2?"

#### Domain 3: "Petr Fiala's specific actions in EU Council" (4 errors)
- **Error types**: PC→FC(4)
- **Languages**: uk(1), sr(1), el(1), ga(1)
- **Pattern**: Consistent over-prediction to FC
- **Insight**: Model may be giving too much credit for partial answers in this domain

#### Domain 4: "Exact quantitative pattern" (4 errors)
- **Error types**: PC→NC(2), NC→PC(2)
- **Languages**: en(1), pt(3)
- **Pattern**: Symmetric NC/PC confusion, Portuguese-heavy

#### Domain 5: "RAS-driven signalling" (4 errors)
- **Error types**: FC→PC(2), PC→FC(2)
- **Languages**: en(1), ga(1), hu(1), de(1)
- **Pattern**: Symmetric FC/PC confusion

**Summary**: 5 domains account for **26/59 errors (44.1%)**. These domains have specific rubric interpretation challenges that could benefit from targeted calibration.

---

## 5. Short Answer Error Analysis

### Short Answer Errors (<100 characters)

- **Total**: 18/59 errors (30.5%)
- **Distribution**:
  - PC→NC: 2 errors
  - NC→PC: 4 errors
  - PC→FC: 6 errors
  - FC→PC: 6 errors

### Examples of Short FC→PC Errors (Model Under-Predicts)

1. **ID 30334 (el)**: Answer = "Μίλοζ Ζέμαν" (11 chars)
   - True: FC, Predicted: PC

2. **ID 43047 (en)**: Answer = "Their influence on surface signalling." (38 chars)
   - True: FC, Predicted: PC

3. **ID 68954 (ga)**: Answer = "A n-tionchar ar chomharthaíocht ar dhromchla." (45 chars)
   - True: FC, Predicted: PC

**Pattern**: The model appears to **penalize very short answers**, predicting PC instead of FC even when the short answer is fully correct according to the rubric.

---

## 6. Actionable Patterns for Improvement

### Priority 1: Address Irish (ga) NC→PC Over-Prediction

**Problem**: 13/22 Irish errors are NC→PC (model over-predicts partial credit)

**Potential Solutions**:
1. **Language-specific threshold calibration**: Lower the NC→PC threshold for Irish by ~5-10%
2. **Irish translation quality**: Review if translation features are introducing bias
3. **Per-language KDE bandwidth**: Use separate KDE bandwidths for Irish vs other languages
4. **Sample weighting**: Increase weight on correctly-labeled Irish NC samples

**Expected Impact**: Could fix 3-5 errors, improving QWK by ~0.001-0.002

### Priority 2: Domain-Specific Calibration

**Problem**: 5 domains account for 44% of errors (26/59)

**Potential Solutions**:
1. **Domain-aware confidence adjustment**: Already implemented in push_qwk_higher.py (STEP 7), but could be strengthened
2. **Domain-specific thresholds**: Learn per-domain PC boundaries using domain OOF performance
3. **Domain clustering**: Group similar rubrics and apply cluster-level calibration

**Expected Impact**: Could fix 5-8 errors in high-error domains, improving QWK by ~0.002-0.003

### Priority 3: Short Answer Calibration

**Problem**: 18/59 errors on answers <100 chars, with FC→PC being common

**Potential Solutions**:
1. **Length-aware thresholds**: Lower PC→FC threshold for short answers (<100 chars)
2. **Feature engineering**: Add explicit "short but complete" features
3. **Confidence boosting**: Increase FC confidence when answer length matches rubric expectations

**Expected Impact**: Could fix 3-4 short answer errors, improving QWK by ~0.001

### Priority 4: Label Noise Handling

**Problem**: 26/59 errors are on suspected noisy labels

**Potential Solutions**:
1. **Manual review**: Review the 26 suspected noise errors for potential re-labeling
2. **Exclude from training**: Already implemented via sample weighting (weight=0.3 for suspicious)
3. **Consensus labeling**: Use model consensus + cleanlab to identify strongest noise candidates

**Expected Impact**: If 10-15 of the 26 noise errors are actually mislabeled, correcting them would improve QWK by ~0.003-0.005

### Priority 5: Boundary Confusion (NC↔PC and FC↔PC)

**Problem**: 66% of errors are boundary errors

**Potential Solutions**:
1. **Multi-threshold optimization**: Already implemented, but could use finer grid search
2. **Confidence-based boundary correction**: Already implemented, but could adjust thresholds
3. **Ordinal regression**: Consider using ordinal regression loss to better respect label ordering

**Expected Impact**: Marginal improvements (~0.0005-0.001) as this is already well-optimized

---

## 7. Specific Error Examples

### Example 1: Irish NC→PC Over-Prediction

**ID**: 96590 (ga, NC→PC)
- **Question**: "Ag cur san áireamh an comhthéacs stairiúil den bhallraíocht san Aontais Eorpaigh ag an bPoblacht na Seice agus an éagsúlacht reatha i gcoinne na bpáir..."
- **Answer** (340 chars): "Fágann an euro ina ábhar conspóideach toisc go bhfuil an Poblacht na Seice tar éis na critéir comhtháite Maastricht a chomhlíonadh. Tacaíonn ODS le glacadh an euro chun trád a mhéadú, ach tá imní ag S..."
- **True Label**: NC (0)
- **Predicted**: PC (1)
- **Analysis**: Long answer that doesn't fully address the question, but model gives partial credit. Possibly a noisy label.

### Example 2: Short Answer FC→PC Under-Prediction

**ID**: 30334 (el, FC→PC)
- **Answer** (11 chars): "Μίλοζ Ζέμαν"
- **True Label**: FC (2)
- **Predicted**: PC (1)
- **Analysis**: Very short answer (just a name), but it's fully correct according to rubric. Model penalizes short answers.

### Example 3: Domain-Specific Symmetric Confusion

**ID**: 50398 (sr, PC→NC)
- **Answer**: "Pustiti da potone i raspadne se? Može li korištenje više proizvoda od algi smanjiti emisije CO2?"
- **True Label**: PC (1)
- **Predicted**: NC (0)
- **Domain**: "Study answered 'no' questions" (4 errors, symmetric PC↔NC confusion)
- **Analysis**: This domain has systematic confusion between PC and NC across multiple languages.

---

## 8. Test Set Error Analysis

**22 test set errors** (37.3% of all errors) - cannot be analyzed with cleanlab

| Error Type | Count | Top Languages |
|------------|-------|---------------|
| **NC→PC** | 11 | ga(8), uk(1), en(1), el(1) |
| **FC→PC** | 7 | uk(2), en(2), ga(2), sv(1) |
| **PC→FC** | 3 | ga(3) |
| **PC→NC** | 1 | hu(1) |

**Key Pattern**: Irish (ga) dominates test set NC→PC errors (8/11 = 73%), reinforcing the Irish over-prediction issue.

---

## 9. Recommendations

### Immediate Actions (Likely to improve QWK by 0.002-0.005)

1. **Irish-specific threshold adjustment**: Lower NC→PC threshold for Irish language by tuning language-specific weights
2. **Review top 10 suspected noise labels**: Manually inspect and potentially re-label the 10 lowest-quality errors
3. **Strengthen domain-aware calibration**: Increase confidence adjustment magnitude for the 5 high-error domains

### Medium-Term Improvements (Potential 0.001-0.003 gain)

1. **Short answer calibration**: Add length-aware threshold adjustments
2. **Per-language KDE bandwidths**: Optimize separate bandwidths for Irish vs other languages
3. **Domain clustering**: Group similar rubrics and apply cluster-level calibration

### Long-Term Research

1. **Ordinal regression**: Explore ordinal regression models that better respect label ordering
2. **Multi-task learning**: Joint training on related tasks (e.g., binary NC/PC and PC/FC classifiers)
3. **Active learning**: Target data collection for high-error domains and Irish language

---

## 10. Conclusion

The QWK=0.990 system achieves excellent performance with only 59 errors (1.42% error rate). However, clear patterns emerge:

1. **Nearly half the errors (44%)** may be due to label noise rather than model failures
2. **Irish language** is a significant outlier with 5x higher error rate, primarily NC→PC over-prediction
3. **Domain concentration** suggests rubric-specific interpretation challenges in 5 key domains
4. **Boundary confusion** (NC↔PC, FC↔PC) accounts for 66% of errors, with short answers being particularly problematic

**Estimated improvement potential**: Implementing the Priority 1-3 recommendations could reduce errors from 59 to ~45-50, potentially pushing QWK from 0.990 to **0.993-0.995**.

---

## Appendix: File Locations

- **Predictions**: `/Users/davidcondrey/workspace_local/panclef/sensemaking/submissions/push_qwk_submission.json`
- **Ground Truth**: `/Users/davidcondrey/workspace_local/panclef/sensemaking/../sensemaking-2026-data/devset/dev.rubric.json`
- **Analysis Script**: `/Users/davidcondrey/workspace_local/panclef/sensemaking/push_qwk_higher.py`
- **Error Details**: `/Users/davidcondrey/workspace_local/panclef/sensemaking/error_analysis.json`
