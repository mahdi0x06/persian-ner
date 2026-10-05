# Persian Named Entity Recognition with ParsBERT

A reproducible token-classification pipeline for Persian named entity recognition (NER), built with **ParsBERT**, Hugging Face Transformers, PyTorch, and the **PEYMA–ARMAN Mixed** dataset.

The project covers the complete training path from word-level BIO labels to subword-aligned model inputs, dynamic batching, fine-tuning, and entity-level evaluation.

## Results

The best recorded validation result after three training epochs was:

| Metric | Validation score |
|---|---:|
| Precision | 0.8786 |
| Recall | 0.9125 |
| F1-score | **0.8952** |

Metrics are computed with `seqeval` after removing ignored special and continuation tokens.

## Dataset

The project loads the public Hugging Face dataset:

- **Dataset:** [AliFartout/PEYMA-ARMAN-Mixed](https://huggingface.co/datasets/AliFartout/PEYMA-ARMAN-Mixed)
- **Task:** Persian token classification
- **Tagging scheme:** BIO
- **Splits used:** train and validation during training; the dataset test split is reserved for final evaluation

The label space contains 21 tags covering dates, events, facilities, locations, money, organizations, percentages, persons, products, times, and the outside label `O`.

## Model

The token-classification model is initialized from:

- **Checkpoint:** [HooshvareLab/bert-base-parsbert-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased)
- **Architecture:** ParsBERT with a token-classification head
- **Number of labels:** 21

The classification head is fine-tuned together with the pretrained transformer.

## Preprocessing and Label Alignment

The dataset provides tokens and one NER label per original word. Because the tokenizer may split a word into multiple subword tokens, the labels must be aligned with the tokenized sequence.

This implementation:

1. tokenizes pre-split words with `is_split_into_words=True`;
2. assigns the original label to the first subword of each word;
3. assigns `-100` to continuation subwords and special tokens;
4. lets PyTorch ignore those `-100` positions when computing the loss.

This prevents duplicated supervision for words that are split into multiple subwords.

## Training Configuration

| Setting | Value |
|---|---:|
| Batch size | 16 |
| Learning rate | 5e-5 |
| Epochs | 3 |
| Optimizer | AdamW |
| Random seed | 42 |
| Checkpoint selection | Best validation F1 |

CUDA is used automatically when available; otherwise, the code falls back to CPU.

The best model and its tokenizer are saved together to:

```text
outputs/best_model/
```

## Repository Structure

```text
persian-ner/
├── src/
│   ├── batch.py          # Dynamic token-classification batching
│   ├── evaluate.py       # Entity-level precision, recall, and F1
│   ├── forward.py        # Forward-pass and prediction inspection
│   ├── model.py          # ParsBERT token-classification model
│   ├── overfit_test.py   # Small-sample implementation sanity check
│   ├── preprocess.py     # Tokenization and label alignment
│   └── train.py          # Training, validation, and checkpointing
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/mahdi0x06/persian-ner.git
cd persian-ner

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The model and dataset are downloaded from Hugging Face when the scripts are run for the first time.

## Usage

Run the preprocessing and alignment checks:

```bash
python -m src.preprocess
```

Inspect dynamic batching:

```bash
python -m src.batch
```

Run a forward-pass sanity check:

```bash
python -m src.forward
```

Verify that the pipeline can overfit a tiny sample:

```bash
python -m src.overfit_test
```

Train the full model:

```bash
python -m src.train
```

Training prints the loss and validation precision, recall, and F1 for every epoch and saves the checkpoint with the highest validation F1.

## Evaluation

Evaluation is performed at the entity level using `seqeval`.

Before metric computation:

- positions labeled `-100` are removed;
- numeric IDs are mapped back to BIO tags;
- dataset-style tags such as `B_PER` are converted to the `B-PER` format expected by `seqeval`.

## Reproducibility Notes

Python, NumPy, and PyTorch random seeds are set to 42. CUDA seeds are also set when a GPU is available.

Some low-level GPU operations can still be nondeterministic unless stricter PyTorch deterministic settings are enabled.

## Limitations and Future Work

- Report final performance on the held-out test split.
- Add an inference script for raw Persian text.
- Save detailed per-entity metrics and error-analysis examples.
- Add configuration files and pinned environment versions.
- Compare ParsBERT with newer Persian or multilingual encoders.
- Explore class imbalance and rare-entity performance.

## License

This project is released under the [MIT License](LICENSE).

