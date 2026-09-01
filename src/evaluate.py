import torch
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from src.model import ID2LABEL


def evaluate_model(model, data_loader, device):
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )

            labels = batch["labels"]

            for prediction, label in zip(predictions, labels):
                sentence_predictions = []
                sentence_labels = []

                for predicted_id, true_id in zip(
                    prediction.tolist(),
                    label.tolist(),
                ):
                    if true_id == -100:
                        continue

                    sentence_predictions.append(
                        convert_tag_format(
                            ID2LABEL[predicted_id]
                        )
                    )
    
                    sentence_labels.append(
                        convert_tag_format(
                            ID2LABEL[true_id]
                        )
                    )

                predicted_labels.append(sentence_predictions)
                true_labels.append(sentence_labels)

    precision = precision_score(
        true_labels,
        predicted_labels,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
    )

    return precision, recall, f1

def convert_tag_format(tag):
    if tag == "O":
        return tag

    return tag.replace("_", "-")