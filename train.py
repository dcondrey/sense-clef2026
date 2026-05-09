#!/usr/bin/env python3
"""
Train all models for TIRA Docker container.

Run this ONCE before building the Docker image. It:
1. Translates Irish items
2. Extracts V3 features for all devset items
3. Fine-tunes DeBERTa with CORAL head on FULL devset
4. Trains LGB + MLP ensemble on all features
5. Calibrates PoE weights and KDE bandwidths
6. Saves everything to models/tira/ for the inference script

Supports both rubric (3-class) and simple (5-class) tracks.

Usage:
    python train.py --data path/to/dev.rubric.json --output models/
    python train.py --track simple --data path/to/dev.simple.json --output models/
"""

import os
import sys
import json
import gc
import argparse
import pickle
import logging
import warnings

import numpy as np

warnings.filterwarnings('ignore')

import torch

_FORCE_CPU = os.environ.get('SENSE_FORCE_CPU', '0') == '1'
if _FORCE_CPU:
    torch.backends.mps.is_available = lambda: False

import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb

from sense.models import CORALHead, DeBERTaRubricScorer, RubricDataset, SimpleDataset
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


def coral_loss(logits, labels, n_classes=3):
    n_thresholds = n_classes - 1
    loss = 0
    for k in range(n_thresholds):
        target = (labels > k).float()
        loss += F.binary_cross_entropy_with_logits(logits[:, k], target)
    return loss / n_thresholds


def soft_qwk_loss(probs, labels, n_classes=3):
    y_onehot = F.one_hot(labels, n_classes).float()
    C = torch.mm(y_onehot.t(), probs)
    classes = torch.arange(n_classes, device=probs.device, dtype=torch.float)
    W = ((classes.unsqueeze(0) - classes.unsqueeze(1)) ** 2) / ((n_classes - 1) ** 2)
    N = labels.shape[0]
    observed = (C * W).sum() / N
    row_sums = C.sum(dim=1)
    col_sums = C.sum(dim=0)
    expected = (torch.outer(row_sums, col_sums) * W).sum() / (N * N)
    return observed / (expected + 1e-8)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def train_deberta(data, model_name='cross-encoder/nli-deberta-v3-base',
                  n_epochs=12, batch_size=8, lr=2e-5, save_dir='models/deberta',
                  n_classes=3, dataset_cls=None):
    """Fine-tune DeBERTa on full devset, save model weights."""
    if dataset_cls is None:
        dataset_cls = RubricDataset
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaRubricScorer(model_name, n_classes=n_classes, dropout=0.1)

    # Training datasets need labels, use padded collation
    class LabeledDataset(dataset_cls):
        def __getitem__(self, idx):
            result = super().__getitem__(idx)
            item = self.data[idx]
            label = item['label']
            result['label'] = torch.tensor(label, dtype=torch.long)
            # Pad to max_length for training stability
            encoding = self.tokenizer(
                ' '.join(item.get('input', item).get('answer', '').split()[:150]),
                self._get_text_b(item),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result['input_ids'] = encoding['input_ids'].squeeze(0)
            result['attention_mask'] = encoding['attention_mask'].squeeze(0)
            return result

        def _get_text_b(self, item):
            inp = item.get('input', item)
            if 'rubrics' in inp:
                fc = ' '.join(inp.get('rubrics', {}).get('FC', '').split()[:120])
                nc = ' '.join(inp.get('rubrics', {}).get('NC', '').split()[:80])
                return f"{fc} [SEP] {nc}"
            else:
                q = ' '.join(inp.get('question', '').split()[:60])
                c = ' '.join(inp.get('context', '').split()[:400])
                return f"{q} [SEP] {c}"

    dataset = LabeledDataset(data, tokenizer, max_length=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    labels = np.array([d['label'] for d in data])
    logger.info(f"  Training DeBERTa on {len(data)} items for {n_epochs} epochs...")
    logger.info(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss_coral = coral_loss(logits, batch_labels, n_classes=n_classes)
            probs = model.coral_head.predict_proba(logits)
            loss_qwk = soft_qwk_loss(probs, batch_labels, n_classes=n_classes)
            loss = loss_coral + 0.3 * loss_qwk

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        logger.info(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump({'model_name': model_name, 'n_classes': n_classes, 'dropout': 0.1}, f)
    logger.info(f"  Saved DeBERTa model to {save_dir}/")
    return model, tokenizer


def predict_deberta(model, tokenizer, data, batch_size=16, device=None, dataset_cls=None):
    """Get DeBERTa CORAL scores for a dataset."""
    from sense.models import dynamic_collate_fn
    if device is None:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    if dataset_cls is None:
        dataset_cls = RubricDataset
    model.eval()
    model.to(device)

    dataset = dataset_cls(data, tokenizer, max_length=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=dynamic_collate_fn)

    all_scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            scores = model.coral_head.predict_scores(logits)
            all_scores.extend(scores.cpu().numpy())

    return np.array(all_scores)


def translate_irish(data):
    """Translate Irish answers to English (returns modified copy)."""
    return translate_items(data)


def main():
    parser = argparse.ArgumentParser(description='SENSE: Train all models for TIRA')
    parser.add_argument('--data', required=True, help='Path to devset JSON file')
    parser.add_argument('--output', default='models/', help='Output directory for models')
    parser.add_argument('--track', default='rubric', choices=['rubric', 'simple'],
                        help='Track: rubric (3-class) or simple (5-class)')
    parser.add_argument('--deberta-epochs', type=int, default=12)
    parser.add_argument('--skip-deberta', action='store_true',
                       help='Skip DeBERTa fine-tuning (if weights already exist)')
    args = parser.parse_args()

    track = args.track
    n_classes = 3 if track == 'rubric' else 5
    max_label = n_classes - 1

    tira_dir = os.path.join(args.output, 'tira')

    if track == 'simple':
        features_save = os.path.join(tira_dir, 'simple_X_all.npy')
        labels_save = os.path.join(tira_dir, 'simple_y_all.npy')
        deberta_dir = os.path.join(tira_dir, 'simple_deberta')
        deberta_scores_save = os.path.join(tira_dir, 'simple_deberta_scores_all.npy')
        bundle_save = os.path.join(tira_dir, 'simple_ensemble_bundle.pkl')
        DatasetCls = SimpleDataset
    else:
        features_save = os.path.join(tira_dir, 'X_all.npy')
        labels_save = os.path.join(tira_dir, 'y_all.npy')
        deberta_dir = os.path.join(tira_dir, 'deberta')
        deberta_scores_save = os.path.join(tira_dir, 'deberta_scores_all.npy')
        bundle_save = os.path.join(tira_dir, 'ensemble_bundle.pkl')
        DatasetCls = RubricDataset

    os.makedirs(tira_dir, exist_ok=True)

    # Load data
    logger.info("=" * 70)
    logger.info(f"LOADING DATA ({track} track, {n_classes} classes)")
    logger.info("=" * 70)
    with open(args.data) as f:
        all_data = json.load(f)
    labels = np.array([d['label'] for d in all_data])
    logger.info(f"Total items: {len(all_data)}")
    logger.info(f"Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Step 1: Translate Irish items
    logger.info("=" * 70)
    logger.info("TRANSLATING IRISH ITEMS")
    logger.info("=" * 70)
    data_translated = translate_irish(all_data)
    gc.collect()

    # Step 2: Extract features
    logger.info("=" * 70)
    logger.info(f"EXTRACTING FEATURES ({track} track)")
    logger.info("=" * 70)
    logger.info("Loading SBERT model...")
    sbert = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("Loading NLI model...")
    nli = CrossEncoder('cross-encoder/nli-deberta-v3-base', max_length=512)

    extractor = FeatureExtractor(sbert, nli)
    logger.info(f"Extracting features for {len(data_translated)} items...")
    if track == 'simple':
        X_all = extractor.extract_simple_features(data_translated, verbose=True)
        ces_all = compute_simple_ces_features(X_all)
    else:
        X_all = extractor.extract_dataset_features(data_translated, verbose=True)
        ces_all = compute_ces_features(X_all)
    n_features_v3 = X_all.shape[1]
    logger.info(f"V3 features shape: {X_all.shape}")

    X_full = np.hstack([ces_all, X_all])
    logger.info(f"CES + V3 features: {X_full.shape[1]}")

    # Retrieval features (rubric track only)
    answer_embs_norm = None
    question_keys = None
    ret_probs_all = None
    if track == 'rubric':
        logger.info("Computing SBERT answer embeddings for retrieval...")
        answers_for_emb = [d.get('input', d).get('answer', '') for d in data_translated]
        answer_embs = sbert.encode(answers_for_emb, batch_size=64, show_progress_bar=True)
        answer_embs_norm = answer_embs / (np.linalg.norm(answer_embs, axis=1, keepdims=True) + 1e-10)
        question_keys = [get_question_key(d) for d in data_translated]

        logger.info("Computing LOO retrieval features...")
        ret_probs_all, ret_raw_all = retrieval_score(
            data_translated, answer_embs_norm, data_translated, answer_embs_norm, labels,
            sigma=0.1, exclude_self=True
        )
        ret_features = np.hstack([ret_probs_all, ret_raw_all.reshape(-1, 1)])
        X_full = np.hstack([X_full, ret_features])

    n_features = X_full.shape[1]
    logger.info(f"Full features: {n_features}")

    np.save(features_save, X_full)
    np.save(labels_save, labels)

    del sbert, nli, extractor
    gc.collect()

    # Step 3: Fine-tune DeBERTa
    logger.info("=" * 70)
    logger.info("FINE-TUNING DeBERTa")
    logger.info("=" * 70)

    if args.skip_deberta and os.path.exists(os.path.join(deberta_dir, 'model.pt')):
        logger.info("Skipping DeBERTa training (--skip-deberta, weights exist)")
        with open(os.path.join(deberta_dir, 'config.json')) as f:
            cfg = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(deberta_dir)
        model = DeBERTaRubricScorer(cfg['model_name'], cfg['n_classes'], cfg['dropout'])
        model.load_state_dict(torch.load(os.path.join(deberta_dir, 'model.pt'),
                                          map_location='cpu', weights_only=True))
    else:
        model, tokenizer = train_deberta(
            data_translated, n_epochs=args.deberta_epochs,
            save_dir=deberta_dir, n_classes=n_classes,
            dataset_cls=DatasetCls
        )

    logger.info("  Computing DeBERTa scores for all data...")
    deberta_scores = predict_deberta(model, tokenizer, data_translated,
                                     batch_size=16, dataset_cls=DatasetCls)
    np.save(deberta_scores_save, deberta_scores)
    logger.info(f"  DeBERTa scores range: [{deberta_scores.min():.3f}, {deberta_scores.max():.3f}]")

    del model, tokenizer
    gc.collect()

    # Step 4: Cleanlab noise detection
    logger.info("=" * 70)
    logger.info("CLEANLAB NOISE DETECTION")
    logger.info("=" * 70)
    from cleanlab.rank import get_label_quality_scores

    db_kde_for_cl = score_to_probs_kde(deberta_scores, deberta_scores, labels,
                                        bandwidth=0.23, n_classes=n_classes)
    db_kde_for_cl = np.clip(db_kde_for_cl, 1e-6, 1.0)
    db_kde_for_cl /= db_kde_for_cl.sum(axis=1, keepdims=True)

    label_quality = get_label_quality_scores(labels, db_kde_for_cl, method='self_confidence')
    noise_threshold = 0.4
    suspicious_mask = label_quality < noise_threshold
    logger.info(f"Label quality: min={label_quality.min():.3f}, mean={label_quality.mean():.3f}")
    logger.info(f"Suspicious items (quality < {noise_threshold}): {suspicious_mask.sum()}")

    sample_weights = np.ones(len(labels))
    sample_weights[suspicious_mask] = 0.3
    medium_mask = (label_quality >= noise_threshold) & (label_quality < 0.6)
    sample_weights[medium_mask] = 0.7
    logger.info(f"Weights: full={int(np.sum(sample_weights == 1.0))}, "
                f"reduced={int(np.sum(sample_weights == 0.7))}, low={int(np.sum(sample_weights == 0.3))}")

    # Step 5: Train LGB + MLP ensemble via 5-fold OOF
    logger.info("=" * 70)
    logger.info("TRAINING LGB + MLP ENSEMBLE (noise-aware)")
    logger.info("=" * 70)

    SEEDS = [42, 123, 456, 789, 2024]
    N_FOLDS = 5
    constraints = [1, 1, 1, -1] + [0] * (n_features - 4)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    lgb_oof = np.zeros(len(labels))
    mlp_oof = np.zeros(len(labels))
    lgb_models = []
    mlp_models = []

    logger.info("  Training LGB (with sample weights)...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        fold_weights = sample_weights[train_idx]
        for seed in SEEDS:
            m = lgb.LGBMRegressor(
                max_depth=7, learning_rate=0.05, n_estimators=500,
                num_leaves=31, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                monotone_constraints=constraints,
                random_state=seed, verbose=-1
            )
            m.fit(X_full[train_idx], labels[train_idx], sample_weight=fold_weights)
            lgb_oof[val_idx] += m.predict(X_full[val_idx])
        lgb_oof[val_idx] /= len(SEEDS)
        logger.info(f"    Fold {fold+1}: OOF QWK={qwk(labels[val_idx], np.round(np.clip(lgb_oof[val_idx], 0, max_label)).astype(int)):.4f}")

    for seed in SEEDS:
        m = lgb.LGBMRegressor(
            max_depth=7, learning_rate=0.05, n_estimators=500,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            monotone_constraints=constraints,
            random_state=seed, verbose=-1
        )
        m.fit(X_full, labels, sample_weight=sample_weights)
        lgb_models.append(m)

    logger.info("  Training MLP...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        sc = StandardScaler()
        X_f = sc.fit_transform(X_full[train_idx])
        X_v = sc.transform(X_full[val_idx])
        for seed in SEEDS:
            m = MLPRegressor(
                hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                learning_rate='adaptive', learning_rate_init=0.001, random_state=seed
            )
            m.fit(X_f, labels[train_idx])
            mlp_oof[val_idx] += m.predict(X_v)
        mlp_oof[val_idx] /= len(SEEDS)

    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X_full)
    for seed in SEEDS:
        m = MLPRegressor(
            hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            learning_rate='adaptive', learning_rate_init=0.001, random_state=seed
        )
        m.fit(X_full_scaled, labels)
        mlp_models.append(m)

    combo_oof = (lgb_oof + mlp_oof) / 2
    logger.info(f"  LGB+MLP OOF combo QWK: {qwk(labels, np.round(np.clip(combo_oof, 0, max_label)).astype(int)):.4f}")

    # Step 6: Per-class KDE + PoE calibration
    logger.info("=" * 70)
    logger.info(f"CALIBRATING PoE ({track} track)")
    logger.info("=" * 70)

    bw_values = np.arange(0.08, 0.36, 0.03)
    best_bws = [0.23] * n_classes
    best_poe_qwk = 0
    best_poe_weights = (0.50, 0.50, 0.0)

    if track == 'simple':
        logger.info("  Searching grouped KDE bandwidths + PoE weights (simple)...")
        for bw_boundary in bw_values:
            for bw_interior in bw_values:
                for bw_top in bw_values:
                    bws = [bw_boundary] + [bw_interior] * (n_classes - 2) + [bw_top]
                    db_kde = safe_probs(score_to_probs_kde_perclass(
                        deberta_scores, deberta_scores, labels, bws, n_classes))
                    combo_kde = safe_probs(score_to_probs_kde_perclass(
                        combo_oof, combo_oof, labels, bws, n_classes))
                    for w in np.arange(0.25, 0.80, 0.05):
                        log_p = w * np.log(db_kde + 1e-10) + (1 - w) * np.log(combo_kde + 1e-10)
                        combined = np.exp(log_p)
                        combined /= combined.sum(axis=1, keepdims=True)
                        q = qwk(labels, np.argmax(combined, axis=1))
                        if q > best_poe_qwk:
                            best_poe_qwk = q
                            best_bws = list(bws)
                            best_poe_weights = (w, 1 - w, 0.0)
    else:
        ret_probs_safe = safe_probs(ret_probs_all)
        logger.info("  Searching per-class KDE bandwidths + PoE weights (rubric)...")
        for bw0 in bw_values:
            for bw1 in bw_values:
                for bw2 in bw_values:
                    bws = [bw0, bw1, bw2]
                    db_kde = safe_probs(score_to_probs_kde_perclass(
                        deberta_scores, deberta_scores, labels, bws, n_classes))
                    combo_kde = safe_probs(score_to_probs_kde_perclass(
                        combo_oof, combo_oof, labels, bws, n_classes))
                    for w in np.arange(0.35, 0.75, 0.05):
                        log_p = w * np.log(db_kde + 1e-10) + (1 - w) * np.log(combo_kde + 1e-10)
                        combined = np.exp(log_p)
                        combined /= combined.sum(axis=1, keepdims=True)
                        q = qwk(labels, np.argmax(combined, axis=1))
                        if q > best_poe_qwk:
                            best_poe_qwk = q
                            best_bws = list(bws)
                            best_poe_weights = (w, 1 - w, 0.0)
                    for w_db in np.arange(0.30, 0.65, 0.05):
                        for w_combo in np.arange(0.15, 0.55, 0.05):
                            w_ret = 1.0 - w_db - w_combo
                            if w_ret < 0.02 or w_ret > 0.30:
                                continue
                            log_p = (w_db * np.log(db_kde + 1e-10) +
                                     w_combo * np.log(combo_kde + 1e-10) +
                                     w_ret * np.log(ret_probs_safe + 1e-10))
                            combined = np.exp(log_p)
                            combined /= combined.sum(axis=1, keepdims=True)
                            q = qwk(labels, np.argmax(combined, axis=1))
                            if q > best_poe_qwk:
                                best_poe_qwk = q
                                best_bws = list(bws)
                                best_poe_weights = (w_db, w_combo, w_ret)

    w_db, w_combo, w_ret = best_poe_weights
    logger.info(f"  Best KDE bandwidths: {best_bws}")
    logger.info(f"  Best PoE weights: db={w_db:.2f}, combo={w_combo:.2f}, ret={w_ret:.2f}")
    logger.info(f"  OOF QWK: {best_poe_qwk:.4f}")

    db_kde = safe_probs(score_to_probs_kde_perclass(
        deberta_scores, deberta_scores, labels, best_bws, n_classes))
    combo_kde = safe_probs(score_to_probs_kde_perclass(
        combo_oof, combo_oof, labels, best_bws, n_classes))

    if w_ret > 0 and ret_probs_all is not None:
        ret_probs_safe = safe_probs(ret_probs_all)
        log_p = (w_db * np.log(db_kde + 1e-10) +
                 w_combo * np.log(combo_kde + 1e-10) +
                 w_ret * np.log(ret_probs_safe + 1e-10))
    else:
        log_p = w_db * np.log(db_kde + 1e-10) + w_combo * np.log(combo_kde + 1e-10)
    combined = np.exp(log_p)
    combined /= combined.sum(axis=1, keepdims=True)
    final_preds = np.argmax(combined, axis=1)
    full_qwk = qwk(labels, final_preds)
    full_acc = np.mean(labels == final_preds)
    logger.info(f"  Full devset: QWK={full_qwk:.4f}, Acc={full_acc:.4f}")

    # Step 7: Save all models
    logger.info("=" * 70)
    logger.info("SAVING MODELS")
    logger.info("=" * 70)

    save_bundle = {
        'lgb_models': lgb_models,
        'mlp_models': mlp_models,
        'scaler': final_scaler,
        'poe_weights': list(best_poe_weights),
        'n_features': n_features,
        'n_features_v3': n_features_v3,
        'train_deberta_scores': deberta_scores,
        'train_combo_scores': combo_oof,
        'train_labels': labels,
        'kde_bandwidths': best_bws,
        'track': track,
        'n_classes': n_classes,
    }

    if track == 'rubric':
        save_bundle['train_answer_embs'] = answer_embs_norm
        save_bundle['train_question_keys'] = question_keys
        save_bundle['retrieval_sigma'] = 0.1

    with open(bundle_save, 'wb') as f:
        pickle.dump(save_bundle, f)

    logger.info(f"  Saved ensemble bundle to {bundle_save}")
    logger.info(f"  Saved DeBERTa to {deberta_dir}/")

    logger.info("=" * 70)
    logger.info(f"DONE! {track.upper()} track models saved to {tira_dir}/")
    logger.info(f"Expected QWK on devset: {full_qwk:.4f}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
