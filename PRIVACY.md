# Privacy Policy

## Data Handling

This software processes text data for automated answer assessment. It is designed to run entirely locally with no data transmitted to external services.

### What This Software Does

- Processes input text (questions, answers, rubrics) for quality assessment
- Runs inference using locally-stored machine learning models
- Outputs classification labels (0, 1, 2) corresponding to assessment grades

### What This Software Does NOT Do

- Does not collect, store, or transmit personal information
- Does not send data to external APIs or cloud services during inference
- Does not track usage or collect analytics
- Does not require internet access for inference (all models are bundled)

### Training Data

The models were trained on the CLEF 2026 ELOQUENT Sensemaking development dataset. No personally identifiable information was used in training. See the [task description](https://eloquent-clef.github.io/) for dataset details.

### Third-Party Models

This software uses the following pre-trained models, which are downloaded during setup:

- `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-BERT)
- `cross-encoder/nli-deberta-v3-base` (NLI cross-encoder)
- `Helsinki-NLP/opus-mt-ga-en` (Irish-English translation)

These models are loaded locally and no data is sent to their providers during inference.

## Contact

For privacy concerns, please open an issue on the GitHub repository.
