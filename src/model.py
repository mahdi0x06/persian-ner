from transformers import AutoModelForTokenClassification

from src.preprocess import MODEL_NAME


LABEL_NAMES = [
    "B_DAT",
    "B_LOC",
    "B_MON",
    "B_ORG",
    "B_PCT",
    "B_PER",
    "B_TIM",
    "I_DAT",
    "I_LOC",
    "I_MON",
    "I_ORG",
    "I_PCT",
    "I_PER",
    "I_TIM",
    "B_FAC",
    "I_FAC",
    "B_EVE",
    "I_EVE",
    "B_PRO",
    "I_PRO",
    "O",
]

NUM_LABELS = len(LABEL_NAMES)

ID2LABEL = {
    index: label
    for index, label in enumerate(LABEL_NAMES)
}

LABEL2ID = {
    label: index
    for index, label in enumerate(LABEL_NAMES)
}


def get_model():
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    return model


if __name__ == "__main__":
    model = get_model()

    print(model)

    print("\nClassification head:")
    print(model.classifier)