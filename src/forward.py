import torch
from datasets import load_dataset

from src.batch import data_collator
from src.model import ID2LABEL, get_model
from src.preprocess import (
    DATASET_NAME,
    preprocess_dataset,
    tokenizer,
)

dataset = load_dataset(DATASET_NAME)
tokenized_dataset = preprocess_dataset(dataset)

samples = [
    tokenized_dataset["train"][0], 
    tokenized_dataset["train"][1],
]

batch = data_collator(samples)

model = get_model()
model.eval

with torch.no_grad():
    outputs = model(**batch)

print("Loss:")
print(outputs.loss)

print("\nLogits shape:")
print(outputs.logits.shape)

predictions = torch.argmax(
    outputs.logits,
    dim=-1,
)

print("\nPredictions shape:")
print(predictions.shape)

tokens = tokenizer.convert_ids_to_tokens(
    batch["input_ids"][0].tolist()
)

print("\nFirst sample predictions:")

for token, true_label_id, predicted_label_id in zip(
    tokens,
    batch["labels"][0].tolist(),
    predictions[0].tolist(),
):
    if true_label_id == -100:
        continue

    true_label = ID2LABEL[true_label_id]
    predicted_label = ID2LABEL[predicted_label_id]

    print(
        f"{token:15} "
        f"true={true_label:6} "
        f"pred={predicted_label:6}"
    )