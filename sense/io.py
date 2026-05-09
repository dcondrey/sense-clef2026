"""
I/O utilities for SENSE.

Handles TIRA-compatible input loading, output writing,
track detection, and model loading.
"""

import os
import json
import pickle
import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_input_data(input_dir):
    """Load input data from TIRA input directory."""
    candidates = ['input.jsonl', 'dataset.jsonl', 'data.jsonl', 'test.jsonl',
                   'input.json', 'dataset.json', 'data.json', 'test.json']

    input_file = None
    for fname in candidates:
        fpath = os.path.join(input_dir, fname)
        if os.path.exists(fpath):
            input_file = fpath
            break

    if input_file is None:
        for f in sorted(os.listdir(input_dir)):
            if f.endswith('.jsonl') or f.endswith('.json'):
                input_file = os.path.join(input_dir, f)
                break

    if input_file is None:
        raise FileNotFoundError(f"No input file found in {input_dir}")

    logger.info(f"Loading input from: {input_file}")

    samples = []
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            samples = data if isinstance(data, list) else [data]

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def write_output(output_dir, predictions):
    """Write predictions to TIRA output directory."""
    os.makedirs(output_dir, exist_ok=True)

    json_file = os.path.join(output_dir, 'predictions.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    jsonl_file = os.path.join(output_dir, 'predictions.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    logger.info(f"Wrote {len(predictions)} predictions to {output_dir}/")


def detect_track(samples):
    """Detect whether input is rubric track (0-2) or simple track (0-4)."""
    first_inp = samples[0].get('input', samples[0])
    has_rubrics = 'rubrics' in first_inp
    if has_rubrics:
        return 'rubric', 3
    return 'simple', 5


def load_models(model_dir, track, n_classes):
    """Load all pre-trained models from model directory.

    Returns a dict with all model components needed for inference.
    """
    if track == 'simple':
        bundle_name = 'simple_ensemble_bundle.pkl'
        models_json_name = 'simple_deberta_models.json'
        deberta_prefix = 'simple_deberta'
    else:
        bundle_name = 'ensemble_bundle.pkl'
        models_json_name = 'deberta_models.json'
        deberta_prefix = 'deberta'

    bundle_path = os.path.join(model_dir, bundle_name)
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)

    lgb_models = bundle['lgb_models']
    mlp_models = bundle['mlp_models']
    scaler = bundle['scaler']
    poe_weights = bundle.get('poe_weights', [0.5, 0.5, 0.0])
    if len(poe_weights) == 2:
        poe_weights = list(poe_weights) + [0.0]
    train_deberta_scores = bundle['train_deberta_scores']
    train_combo_scores = bundle['train_combo_scores']
    train_labels = bundle['train_labels']
    kde_bandwidths = bundle.get('kde_bandwidths', [0.23] * n_classes)
    n_features = bundle['n_features']
    train_answer_embs = bundle.get('train_answer_embs')
    train_question_keys = bundle.get('train_question_keys')
    retrieval_sigma = bundle.get('retrieval_sigma', 0.1)
    language_calibration = bundle.get('language_calibration', {})

    deberta_model_configs = []
    deberta_model_weights = None
    models_json = os.path.join(model_dir, models_json_name)
    if os.path.exists(models_json):
        with open(models_json) as f:
            models_cfg = json.load(f)
        if isinstance(models_cfg, dict):
            model_names = models_cfg['models']
            deberta_model_weights = np.array(models_cfg.get('weights', []))
            if len(deberta_model_weights) == 0:
                deberta_model_weights = None
        else:
            model_names = models_cfg
        for model_name in model_names:
            safe = model_name.split('/')[-1].replace('-', '_')
            d = os.path.join(model_dir, f'{deberta_prefix}_{safe}')
            deberta_model_configs.append((model_name, d))
    else:
        deberta_dir = os.path.join(model_dir, deberta_prefix)
        deberta_model_configs.append((None, deberta_dir))

    logger.info(f"  Loaded {len(lgb_models)} LGB + {len(mlp_models)} MLP models")
    logger.info(f"  PoE weights: db={poe_weights[0]:.2f}, combo={poe_weights[1]:.2f}, ret={poe_weights[2]:.2f}")
    logger.info(f"  KDE bandwidths: {kde_bandwidths}")

    return {
        'lgb_models': lgb_models,
        'mlp_models': mlp_models,
        'scaler': scaler,
        'poe_weights': poe_weights,
        'train_deberta_scores': train_deberta_scores,
        'train_combo_scores': train_combo_scores,
        'train_labels': train_labels,
        'kde_bandwidths': kde_bandwidths,
        'n_features': n_features,
        'train_answer_embs': train_answer_embs,
        'train_question_keys': train_question_keys,
        'retrieval_sigma': retrieval_sigma,
        'language_calibration': language_calibration,
        'deberta_model_configs': deberta_model_configs,
        'deberta_model_weights': deberta_model_weights,
    }
