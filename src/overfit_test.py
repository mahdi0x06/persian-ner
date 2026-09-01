import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset

from src.batch import data_collator
from src.evaluate import evaluate_model
from src.model import get_model
from src.preprocess import DATASET_NAME, preprocess_dataset


BATCH_SIZE = 2
LEARNING_RATE = 5e-5
EPOCHS = 20


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


dataset = load_dataset(DATASET_NAME)
tokenized_dataset = preprocess_dataset(dataset)


small_dataset = tokenized_dataset["train"].select(
    range(10)
)


train_loader = DataLoader(
    small_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)


model = get_model()
model.to(device)


optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


model.train()


for epoch in range(EPOCHS):

    total_loss = 0

    for batch in train_loader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Loss: {avg_loss:.4f}"
    )


precision, recall, f1 = evaluate_model(
    model,
    train_loader,
    device,
)


print("\nOverfit result:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")