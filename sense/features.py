"""
V3 feature extraction for SENSE.

Extracts 73 features (rubric track) or 42 features (simple track)
using SBERT embeddings, NLI cross-encoder scores, structural analysis,
compression distance, rubric keywords, and cross-lingual signals.
"""

import re
import zlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract V3 features from raw data items."""

    def __init__(self, sbert_model, nli_model):
        self.sbert = sbert_model
        self.nli = nli_model

    def get_embedding(self, text, max_length=512):
        words = text.split()[:max_length]
        text = ' '.join(words)
        emb = self.sbert.encode(text, show_progress_bar=False)
        return emb / (np.linalg.norm(emb) + 1e-8)

    def cosine_similarity(self, emb1, emb2):
        return float(np.dot(emb1, emb2))

    def get_nli_scores(self, premise, hypothesis):
        try:
            premise = ' '.join(premise.split()[:256])
            hypothesis = ' '.join(hypothesis.split()[:128])
            scores = self.nli.predict([(premise, hypothesis)])
            if isinstance(scores, np.ndarray) and len(scores.shape) > 1:
                return (float(scores[0][0]), float(scores[0][1]), float(scores[0][2]))
            return (0.0, 0.0, float(scores[0]))
        except Exception:
            return (0.33, 0.34, 0.33)

    def extract_features(self, context, question, answer, rubrics):
        """Extract V3 features for one rubric-track item."""
        features = []
        fc = rubrics.get('FC', '')[:500]
        pc = rubrics.get('PC', '')[:500]
        nc = rubrics.get('NC', '')[:500]

        answer_emb = self.get_embedding(answer)
        context_emb = self.get_embedding(context[:2000])
        question_emb = self.get_embedding(question)
        fc_emb = self.get_embedding(fc)
        pc_emb = self.get_embedding(pc)
        nc_emb = self.get_embedding(nc)

        sim_fc = self.cosine_similarity(answer_emb, fc_emb)
        sim_pc = self.cosine_similarity(answer_emb, pc_emb)
        sim_nc = self.cosine_similarity(answer_emb, nc_emb)
        features.extend([sim_fc, sim_pc, sim_nc])
        features.extend([sim_fc - sim_nc, sim_fc - sim_pc, sim_nc - sim_pc])
        features.append(sim_fc - 0.5 * (sim_nc + sim_pc))

        sim_context = self.cosine_similarity(answer_emb, context_emb)
        sim_question = self.cosine_similarity(answer_emb, question_emb)
        features.extend([sim_context, sim_question])

        fc_ctx = self.cosine_similarity(fc_emb, context_emb)
        nc_ctx = self.cosine_similarity(nc_emb, context_emb)
        pc_ctx = self.cosine_similarity(pc_emb, context_emb)
        features.extend([fc_ctx, nc_ctx, pc_ctx])

        fc_q = self.cosine_similarity(fc_emb, question_emb)
        nc_q = self.cosine_similarity(nc_emb, question_emb)
        features.extend([fc_q, nc_q])

        ctx_ans = self.get_nli_scores(context[:1000], answer)
        features.extend(ctx_ans)
        features.append(ctx_ans[2] - ctx_ans[0])

        fc_ans = self.get_nli_scores(fc, answer)
        nc_ans = self.get_nli_scores(nc, answer)
        pc_ans = self.get_nli_scores(pc, answer)
        features.extend(fc_ans)
        features.extend(nc_ans)
        features.extend(pc_ans)

        ans_fc = self.get_nli_scores(answer, fc)
        ans_nc = self.get_nli_scores(answer, nc)
        ans_pc = self.get_nli_scores(answer, pc)
        features.extend([ans_fc[2], ans_nc[2], ans_pc[2]])
        features.extend([ans_fc[0], ans_nc[0], ans_pc[0]])
        features.append(fc_ans[2] - nc_ans[2])
        features.append(fc_ans[2] - pc_ans[2])
        features.append(ans_fc[2] - ans_nc[2])

        features.extend(_structural_features(answer))
        features.extend(_compression_features(answer, fc, nc, pc, context))
        features.extend(_rubric_keyword_features(rubrics))
        features.extend(_crosslingual_features(answer))

        return np.array(features, dtype=np.float32)

    def extract_dataset_features(self, data, verbose=True):
        """Extract V3 features for all items using batched SBERT and NLI."""
        n = len(data)

        logger.info("  Preparing texts for batch encoding...")
        sbert_texts = []
        nli_pairs = []
        items_data = []

        for item in data:
            inp = item.get('input', item)
            context = inp.get('context', '')
            question = inp.get('question', '')
            answer = inp.get('answer', '')
            rubrics = inp.get('rubrics', {})
            fc = rubrics.get('FC', '')[:500]
            pc = rubrics.get('PC', '')[:500]
            nc = rubrics.get('NC', '')[:500]

            sbert_texts.extend([
                ' '.join(answer.split()[:512]),
                ' '.join(context[:2000].split()[:512]),
                ' '.join(question.split()[:512]),
                ' '.join(fc.split()[:512]),
                ' '.join(pc.split()[:512]),
                ' '.join(nc.split()[:512]),
            ])

            ctx_trunc = ' '.join(context[:1000].split()[:256])
            ans_hyp = ' '.join(answer.split()[:128])
            fc_prem = ' '.join(fc.split()[:256])
            nc_prem = ' '.join(nc.split()[:256])
            pc_prem = ' '.join(pc.split()[:256])
            ans_prem = ' '.join(answer.split()[:256])
            fc_hyp = ' '.join(fc.split()[:128])
            nc_hyp = ' '.join(nc.split()[:128])
            pc_hyp = ' '.join(pc.split()[:128])

            nli_pairs.extend([
                (ctx_trunc, ans_hyp),
                (fc_prem, ans_hyp),
                (nc_prem, ans_hyp),
                (pc_prem, ans_hyp),
                (ans_prem, fc_hyp),
                (ans_prem, nc_hyp),
                (ans_prem, pc_hyp),
            ])

            items_data.append({
                'context': context, 'answer': answer,
                'rubrics': rubrics, 'fc': fc, 'pc': pc, 'nc': nc,
            })

        logger.info(f"  Batch encoding {len(sbert_texts)} texts with SBERT...")
        all_embs = self.sbert.encode(sbert_texts, batch_size=256, show_progress_bar=True)
        norms = np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8
        all_embs = all_embs / norms

        logger.info(f"  Batch predicting {len(nli_pairs)} NLI pairs...")
        all_nli = self.nli.predict(nli_pairs, batch_size=64, show_progress_bar=True)
        if not isinstance(all_nli, np.ndarray):
            all_nli = np.array(all_nli)
        if all_nli.ndim == 1:
            all_nli = np.column_stack([np.zeros(len(all_nli)), np.zeros(len(all_nli)), all_nli])

        logger.info("  Computing per-item features...")
        all_features = []
        for idx in range(n):
            emb_base = idx * 6
            answer_emb = all_embs[emb_base]
            context_emb = all_embs[emb_base + 1]
            question_emb = all_embs[emb_base + 2]
            fc_emb = all_embs[emb_base + 3]
            pc_emb = all_embs[emb_base + 4]
            nc_emb = all_embs[emb_base + 5]

            nli_base = idx * 7
            ctx_ans = tuple(float(v) for v in all_nli[nli_base])
            fc_ans = tuple(float(v) for v in all_nli[nli_base + 1])
            nc_ans = tuple(float(v) for v in all_nli[nli_base + 2])
            pc_ans = tuple(float(v) for v in all_nli[nli_base + 3])
            ans_fc = tuple(float(v) for v in all_nli[nli_base + 4])
            ans_nc = tuple(float(v) for v in all_nli[nli_base + 5])
            ans_pc = tuple(float(v) for v in all_nli[nli_base + 6])

            d = items_data[idx]
            answer = d['answer']
            context = d['context']
            rubrics = d['rubrics']
            fc, pc, nc = d['fc'], d['pc'], d['nc']

            features = []
            cos = lambda a, b: float(np.dot(a, b))

            sim_fc = cos(answer_emb, fc_emb)
            sim_pc = cos(answer_emb, pc_emb)
            sim_nc = cos(answer_emb, nc_emb)
            features.extend([sim_fc, sim_pc, sim_nc])
            features.extend([sim_fc - sim_nc, sim_fc - sim_pc, sim_nc - sim_pc])
            features.append(sim_fc - 0.5 * (sim_nc + sim_pc))

            sim_context = cos(answer_emb, context_emb)
            sim_question = cos(answer_emb, question_emb)
            features.extend([sim_context, sim_question])

            features.extend([cos(fc_emb, context_emb), cos(nc_emb, context_emb),
                             cos(pc_emb, context_emb)])
            features.extend([cos(fc_emb, question_emb), cos(nc_emb, question_emb)])

            features.extend(ctx_ans)
            features.append(ctx_ans[2] - ctx_ans[0])
            features.extend(fc_ans)
            features.extend(nc_ans)
            features.extend(pc_ans)
            features.extend([ans_fc[2], ans_nc[2], ans_pc[2]])
            features.extend([ans_fc[0], ans_nc[0], ans_pc[0]])
            features.append(fc_ans[2] - nc_ans[2])
            features.append(fc_ans[2] - pc_ans[2])
            features.append(ans_fc[2] - ans_nc[2])

            features.extend(_structural_features(answer))
            features.extend(_compression_features(answer, fc, nc, pc, context))
            features.extend(_rubric_keyword_features(rubrics))
            features.extend(_crosslingual_features(answer))

            all_features.append(np.array(features, dtype=np.float32))
            if verbose and (idx + 1) % 500 == 0:
                logger.info(f"  Features: {idx + 1}/{n}")

        return np.vstack(all_features)

    def extract_simple_features(self, data, verbose=True):
        """Extract features for simple track (no rubrics). 42 features."""
        n = len(data)

        logger.info("  Preparing texts for batch encoding (simple track)...")
        sbert_texts = []
        nli_pairs = []
        items_data = []

        for item in data:
            inp = item.get('input', item)
            context = inp.get('context', '')
            question = inp.get('question', '')
            answer = inp.get('answer', '')

            sbert_texts.extend([
                ' '.join(answer.split()[:512]),
                ' '.join(context[:2000].split()[:512]),
                ' '.join(question.split()[:512]),
            ])

            ctx_trunc = ' '.join(context[:1000].split()[:256])
            ans_hyp = ' '.join(answer.split()[:128])
            q_prem = ' '.join(question.split()[:256])
            ans_prem = ' '.join(answer.split()[:256])
            q_hyp = ' '.join(question.split()[:128])

            nli_pairs.extend([
                (ctx_trunc, ans_hyp),
                (q_prem, ans_hyp),
                (ans_prem, q_hyp),
            ])

            items_data.append({
                'context': context, 'question': question, 'answer': answer,
            })

        logger.info(f"  Batch encoding {len(sbert_texts)} texts with SBERT...")
        all_embs = self.sbert.encode(sbert_texts, batch_size=256, show_progress_bar=True)
        norms = np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8
        all_embs = all_embs / norms

        logger.info(f"  Batch predicting {len(nli_pairs)} NLI pairs...")
        all_nli = self.nli.predict(nli_pairs, batch_size=64, show_progress_bar=True)
        if not isinstance(all_nli, np.ndarray):
            all_nli = np.array(all_nli)
        if all_nli.ndim == 1:
            all_nli = np.column_stack([np.zeros(len(all_nli)), np.zeros(len(all_nli)), all_nli])

        logger.info("  Computing per-item features (simple track)...")
        all_features = []
        for idx in range(n):
            emb_base = idx * 3
            answer_emb = all_embs[emb_base]
            context_emb = all_embs[emb_base + 1]
            question_emb = all_embs[emb_base + 2]

            nli_base = idx * 3
            ctx_ans = tuple(float(v) for v in all_nli[nli_base])
            q_ans = tuple(float(v) for v in all_nli[nli_base + 1])
            ans_q = tuple(float(v) for v in all_nli[nli_base + 2])

            d = items_data[idx]
            answer = d['answer']
            context = d['context']
            question = d['question']

            features = []
            cos = lambda a, b: float(np.dot(a, b))

            sim_context = cos(answer_emb, context_emb)
            sim_question = cos(answer_emb, question_emb)
            sim_q_ctx = cos(question_emb, context_emb)
            features.extend([sim_context, sim_question, sim_q_ctx])

            features.extend(ctx_ans)
            features.append(ctx_ans[2] - ctx_ans[0])
            features.extend(q_ans)
            features.extend([ans_q[2], ans_q[0]])

            features.extend(_structural_features(answer))

            def ncd(x, y):
                if not x or not y:
                    return 1.0
                c_x = len(zlib.compress(x.encode('utf-8'), 9))
                c_y = len(zlib.compress(y.encode('utf-8'), 9))
                c_xy = len(zlib.compress((x + y).encode('utf-8'), 9))
                return (c_xy - min(c_x, c_y)) / (max(c_x, c_y) + 1e-8)

            ncd_ctx = 1 - ncd(answer, context[:1000])
            ncd_q = 1 - ncd(answer, question)
            features.extend([ncd_ctx, ncd_q])
            if answer:
                features.append(len(zlib.compress(answer.encode('utf-8'), 9)) / len(answer.encode('utf-8')))
            else:
                features.append(1.0)
            features.append(ncd_ctx - ncd_q)

            features.extend(_crosslingual_features(answer))

            all_features.append(np.array(features, dtype=np.float32))
            if verbose and (idx + 1) % 500 == 0:
                logger.info(f"  Features: {idx + 1}/{n}")

        return np.vstack(all_features)


def _structural_features(answer):
    """Extract 22 structural features from answer text."""
    features = []
    answer_words = len(answer.split())
    answer_chars = len(answer)
    features.extend([
        answer_words, answer_chars,
        min(1.0, answer_words / 50), min(1.0, answer_words / 100),
        1.0 if answer_words < 10 else 0.0,
        1.0 if answer_words > 40 else 0.0,
        1.0 if answer_words > 60 else 0.0,
    ])

    sentences = len(re.findall(r'[.!?]+', answer))
    features.extend([
        sentences, 1.0 if sentences >= 2 else 0.0,
        1.0 if sentences >= 3 else 0.0, answer_words / (sentences + 1),
    ])

    features.extend([
        answer.count(','), answer.count(':'), answer.count(';'),
        answer.count('('), 1.0 if '"' in answer or "'" in answer else 0.0,
    ])

    numbers = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', answer))
    features.extend([numbers, 1.0 if numbers > 0 else 0.0, 1.0 if numbers >= 2 else 0.0])

    entities = len(re.findall(r'\b[A-Z][a-z\u00e0-\u00ff]+\b', answer))
    features.extend([entities, entities / (answer_words + 1)])

    words_list = answer.lower().split()
    diversity = len(set(words_list)) / max(len(words_list), 1)
    features.append(diversity)

    return features


def _compression_features(answer, fc, nc, pc, context):
    """Extract 7 compression-based features."""
    def ncd(x, y):
        if not x or not y:
            return 1.0
        c_x = len(zlib.compress(x.encode('utf-8'), 9))
        c_y = len(zlib.compress(y.encode('utf-8'), 9))
        c_xy = len(zlib.compress((x + y).encode('utf-8'), 9))
        return (c_xy - min(c_x, c_y)) / (max(c_x, c_y) + 1e-8)

    ncd_fc = 1 - ncd(answer, fc)
    ncd_nc = 1 - ncd(answer, nc)
    ncd_pc = 1 - ncd(answer, pc)
    ncd_ctx = 1 - ncd(answer, context[:1000])
    features = [ncd_fc, ncd_nc, ncd_pc, ncd_ctx]
    features.extend([ncd_fc - ncd_nc, ncd_fc - ncd_pc])

    if answer:
        features.append(len(zlib.compress(answer.encode('utf-8'), 9)) / len(answer.encode('utf-8')))
    else:
        features.append(1.0)

    return features


def _rubric_keyword_features(rubrics):
    """Extract 4 rubric keyword features."""
    fc_lower = rubrics.get('FC', '').lower()
    nc_lower = rubrics.get('NC', '').lower()
    fc_kw = sum(1 for k in ['both', 'all', 'specific', 'explains', 'correct'] if k in fc_lower)
    nc_kw = sum(1 for k in ['does not', 'fails', 'incorrect', 'wrong', 'vague'] if k in nc_lower)
    return [
        fc_kw, nc_kw,
        1.0 if 'both' in fc_lower else 0.0,
        1.0 if 'explains' in fc_lower or 'explanation' in fc_lower else 0.0,
    ]


def _crosslingual_features(answer):
    """Extract 4 cross-lingual signal features."""
    return [
        1.0 if answer.isascii() else 0.0,
        1.0 if re.search(r'[\u011b\u0161\u010d\u0159\u017e\u00fd\u00e1\u00ed\u00e9\u016f\u00fa\u0148]', answer.lower()) else 0.0,
        1.0 if re.search(r'[\u00e4\u00f6\u00fc\u00df]', answer.lower()) else 0.0,
        1.0 if re.search(r'[\u0400-\u04ff]', answer) else 0.0,
    ]
