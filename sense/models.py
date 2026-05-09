"""
Neural model architectures for SENSE.

Contains the DeBERTa CORAL cross-encoder and dataset classes
for both rubric (3-class) and simple (5-class) tracks.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel


class CORALHead(nn.Module):
    """Ordinal regression head using Consistent Rank Logits (CORAL)."""

    def __init__(self, hidden_dim, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        self.n_thresholds = n_classes - 1
        self.fc = nn.Linear(hidden_dim, 1, bias=False)
        init_biases = torch.linspace(1.0, -(n_classes - 2), n_classes - 1)
        self.biases = nn.Parameter(init_biases)

    def forward(self, h):
        proj = self.fc(h)
        logits = proj + self.biases.unsqueeze(0)
        return logits

    def predict_proba(self, logits):
        """Convert CORAL logits to class probabilities."""
        cumprobs = torch.sigmoid(logits)
        probs = torch.zeros(logits.shape[0], self.n_classes, device=logits.device)
        probs[:, 0] = 1 - cumprobs[:, 0]
        for k in range(1, self.n_classes - 1):
            probs[:, k] = cumprobs[:, k - 1] - cumprobs[:, k]
        probs[:, self.n_classes - 1] = cumprobs[:, self.n_classes - 2]
        probs = torch.clamp(probs, min=1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def predict_scores(self, logits):
        """Convert CORAL logits to expected scores."""
        probs = self.predict_proba(logits)
        classes = torch.arange(self.n_classes, device=logits.device, dtype=torch.float)
        return (probs * classes).sum(dim=1)


class DeBERTaRubricScorer(nn.Module):
    """DeBERTa encoder with CORAL ordinal regression head."""

    def __init__(self, model_name, n_classes=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.coral_head = CORALHead(hidden_dim, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.coral_head(cls_output)
        return logits


class RubricDataset(Dataset):
    """Dataset for rubric track: answer + FC [SEP] NC rubrics."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inp = item.get('input', item)
        answer = inp.get('answer', '')
        rubrics = inp.get('rubrics', {})
        fc_rubric = rubrics.get('FC', '')
        nc_rubric = rubrics.get('NC', '')

        answer_trunc = ' '.join(answer.split()[:150])
        fc_trunc = ' '.join(fc_rubric.split()[:120])
        nc_trunc = ' '.join(nc_rubric.split()[:80])

        encoding = self.tokenizer(
            answer_trunc, f"{fc_trunc} [SEP] {nc_trunc}",
            max_length=self.max_length,
            truncation=True,
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }


class SimpleDataset(Dataset):
    """Dataset for simple track: answer + question [SEP] context."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inp = item.get('input', item)
        answer = inp.get('answer', '')
        question = inp.get('question', '')
        context = inp.get('context', '')

        answer_trunc = ' '.join(answer.split()[:150])
        question_trunc = ' '.join(question.split()[:60])
        context_trunc = ' '.join(context.split()[:400])

        encoding = self.tokenizer(
            answer_trunc, f"{question_trunc} [SEP] {context_trunc}",
            max_length=self.max_length,
            truncation=True,
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }


def dynamic_collate_fn(batch):
    """Pad sequences to max length in batch instead of fixed 512."""
    max_len = max(len(b['input_ids']) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        seq_len = len(b['input_ids'])
        input_ids[i, :seq_len] = torch.tensor(b['input_ids'])
        attention_mask[i, :seq_len] = torch.tensor(b['attention_mask'])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}
