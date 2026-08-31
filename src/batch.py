from datasets import load_dataset
from transformers import DataCollatorForTokenClassification

from src.preprocess import (
    DATASET_NAME,
    preprocess_dataset,
    tokenizer,
)


dataset = load_dataset(DATASET_NAME)
tokenized_dataset = preprocess_dataset(dataset)

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)

samples = [
    tokenized_dataset["train"][0],
    tokenized_dataset["train"][1],
]

batch = data_collator(samples)

print("Batch keys:")
print(batch.keys())

print("\nInput IDs shape:")
print(batch["input_ids"].shape)

print("\nAttention mask:")
print(batch["attention_mask"])

print("\nLabels:")
print(batch["labels"])