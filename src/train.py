import os
import random
import time

import numpy as np
import torch

from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.batch import data_collator
from src.evaluate import evaluate_model
from src.model import get_model
from src.preprocess import (
    DATASET_NAME,
    preprocess_dataset,
)


# =====================
# Configuration
# =====================

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 3

CHECKPOINT_PATH = "outputs/best_model"

SEED = 42


# =====================
# Reproducibility
# =====================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =====================
# Device
# =====================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# =====================
# Dataset
# =====================

dataset = load_dataset(DATASET_NAME)

tokenized_dataset = preprocess_dataset(
    dataset
)


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


# =====================
# Model
# =====================

model = get_model()

model.to(device)


optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


best_f1 = 0.0


# =====================
# Training Loop
# =====================

for epoch in range(EPOCHS):

    start_time = time.time()

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )


    model.train()

    total_loss = 0.0


    progress_bar = tqdm(
        train_loader,
        desc="Training",
    )


    for batch in progress_bar:

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


        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    average_loss = (
        total_loss / len(train_loader)
    )


    print(
        f"Train Loss: {average_loss:.4f}"
    )


    # =====================
    # Validation
    # =====================

    precision, recall, f1 = evaluate_model(
        model,
        validation_loader,
        device,
    )


    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )


    elapsed = time.time() - start_time

    print(
        f"Epoch time: {elapsed / 60:.2f} minutes"
    )


    # =====================
    # Save Best Model
    # =====================

    if f1 > best_f1:

        best_f1 = f1

        os.makedirs(
            CHECKPOINT_PATH,
            exist_ok=True,
        )


        print(
            f"Saving best model with F1={f1:.4f}"
        )


        model.save_pretrained(
            CHECKPOINT_PATH
        )


print(
    f"\nBest validation F1: {best_f1:.4f}"
)