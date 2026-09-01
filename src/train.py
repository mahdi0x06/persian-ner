import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.batch import data_collator
from src.evaluate import evaluate_model
from src.model import get_model
from src.preprocess import DATASET_NAME, preprocess_dataset


BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_TEST_BATCHES = 3


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


dataset = load_dataset(DATASET_NAME)
tokenized_dataset = preprocess_dataset(dataset)


train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)


validation_loader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)


model = get_model()
model.to(device)


optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


model.train()

for batch_index, batch in enumerate(train_loader):

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    optimizer.zero_grad()

    outputs = model(**batch)

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    print(
        f"Batch {batch_index + 1} "
        f"| Loss: {loss.item():.4f}"
    )

    if batch_index + 1 == NUM_TEST_BATCHES:
        break


precision, recall, f1 = evaluate_model(
    model,
    validation_loader,
    device,
)


print("\nValidation:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")