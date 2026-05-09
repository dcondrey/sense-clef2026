#!/usr/bin/env python3
"""
Evaluate SENSE pipeline on the sensemaking-2026 dev set.
Computes accuracy, Cohen's kappa, and MAE against ground truth labels.

Usage:
    python evaluate.py -d path/to/devset/ -t rubric
    python evaluate.py -d path/to/devset/ -t both
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


def load_devset(filepath):
    """Load dev set JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_predictions(filepath):
    """Load predictions from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    return {p['id']: p['label'] for p in preds}


def evaluate_track(data, predictions, track_name):
    """Evaluate predictions against ground truth for a single track."""
    logger.info(f"{'=' * 60}")
    logger.info(f"{track_name} Track Evaluation")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total samples: {len(data)}")

    labels = []
    preds = []
    missing = 0

    for item in data:
        item_id = item['id']
        if item_id not in predictions:
            missing += 1
            continue
        labels.append(item['label'])
        preds.append(predictions[item_id])

    if missing > 0:
        logger.warning(f"Missing predictions for {missing} items")

    if not labels:
        logger.error("No matching predictions found")
        return {}

    acc = accuracy_score(labels, preds)
    linear_kappa = cohen_kappa_score(labels, preds, weights='linear')
    quadratic_kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    mae = np.mean(np.abs(np.array(labels) - np.array(preds)))

    logger.info(f"Results:")
    logger.info(f"  Accuracy:                {acc:.4f}")
    logger.info(f"  Linear Cohen's Kappa:    {linear_kappa:.4f}")
    logger.info(f"  Quadratic Cohen's Kappa: {quadratic_kappa:.4f}")
    logger.info(f"  MAE:                     {mae:.4f}")

    return {
        'accuracy': acc,
        'linear_cohen_kappa': linear_kappa,
        'quadratic_cohen_kappa': quadratic_kappa,
        'mae': mae,
        'n_samples': len(labels),
        'n_missing': missing,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate SENSE predictions')
    parser.add_argument('--devset-dir', '-d',
                        default='../sensemaking-2026-data/devset',
                        help='Path to devset directory')
    parser.add_argument('--predictions-dir', '-p',
                        default='predictions',
                        help='Path to predictions directory')
    parser.add_argument('--track', '-t', choices=['simple', 'rubric', 'both'],
                        default='both', help='Which track to evaluate')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file for evaluation summary (JSON)')

    args = parser.parse_args()

    devset_dir = Path(args.devset_dir)
    pred_dir = Path(args.predictions_dir)

    logger.info("=" * 60)
    logger.info("SENSE Dev Set Evaluation")
    logger.info("=" * 60)

    results = {}

    if args.track in ['simple', 'both']:
        simple_path = devset_dir / 'dev.simple.json'
        pred_path = pred_dir / 'predictions_simple.json'
        if simple_path.exists() and pred_path.exists():
            simple_data = load_devset(simple_path)
            simple_preds = load_predictions(pred_path)
            results['simple'] = evaluate_track(simple_data, simple_preds, 'Simple Rating')
        else:
            logger.warning(f"Skipping simple track: missing {simple_path} or {pred_path}")

    if args.track in ['rubric', 'both']:
        rubric_path = devset_dir / 'dev.rubric.json'
        pred_path = pred_dir / 'predictions_rubric.json'
        if rubric_path.exists() and pred_path.exists():
            rubric_data = load_devset(rubric_path)
            rubric_preds = load_predictions(pred_path)
            results['rubric'] = evaluate_track(rubric_data, rubric_preds, 'Rubric-Based Rating')
        else:
            logger.warning(f"Skipping rubric track: missing {rubric_path} or {pred_path}")

    # Summary
    logger.info(f"{'=' * 60}")
    logger.info("Summary")
    logger.info(f"{'=' * 60}")

    for track_name, metrics in results.items():
        logger.info(f"{track_name.upper()} Track:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Summary saved to: {output_path}")


if __name__ == '__main__':
    main()
