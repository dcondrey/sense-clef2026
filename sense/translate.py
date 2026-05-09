"""
Irish language detection and translation.

Only Irish is translated because the multilingual SBERT and NLI models
already handle other European languages well natively. Translating other
languages introduces noise from imperfect translation (especially medical
terminology) without improving feature quality.
"""

import json
import gc
import logging

import torch

logger = logging.getLogger(__name__)


def detect_irish(item):
    """Detect if an item is in Irish using comprehensive keyword matching."""
    if item.get('lang') == 'ga':
        return True
    inp = item.get('input', item)
    answer = inp.get('answer', '')
    answer_lower = ' ' + answer.lower() + ' '
    ga_keywords = [' agus ', ' bhí ', ' tá ', ' seo ', ' atá ', ' nach ',
                   ' ach ', ' mar ', ' chun ']
    if any(w in answer_lower for w in ga_keywords):
        return True
    ga_specific = [' n-', 'fuair ', 'is é ', ' mhná ', ' gcuireann ',
                   ' leigheas', 'comharthaíocht', ' maoiniú', ' ceannaire ',
                   'thionscnamh', ' mbeadh ', ' ilghluaisí', ' córas ',
                   ' ródháileog']
    if any(w in answer_lower for w in ga_specific):
        return True
    return False


def translate_items(data):
    """Translate Irish answers to English. Returns modified data copy."""
    irish_indices = [i for i, item in enumerate(data) if detect_irish(item)]
    if not irish_indices:
        logger.info("No Irish items detected.")
        return data

    logger.info(f"Translating {len(irish_indices)} Irish items...")
    from transformers import MarianMTModel, MarianTokenizer
    mt_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ga-en')
    mt_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ga-en')

    irish_answers = [data[i].get('input', data[i]).get('answer', '') for i in irish_indices]
    translations = []
    for i in range(0, len(irish_answers), 4):
        batch = irish_answers[i:i+4]
        inputs = mt_tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
        with torch.no_grad():
            outputs = mt_model.generate(**inputs, max_length=512, num_beams=4)
        translated = mt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(translated)

    data_out = []
    irish_set = set(irish_indices)
    for i, item in enumerate(data):
        new_item = json.loads(json.dumps(item))
        if i in irish_set:
            idx_in_irish = irish_indices.index(i)
            if 'input' in new_item:
                new_item['input']['answer'] = translations[idx_in_irish]
            else:
                new_item['answer'] = translations[idx_in_irish]
        data_out.append(new_item)

    del mt_model, mt_tokenizer
    gc.collect()
    return data_out
