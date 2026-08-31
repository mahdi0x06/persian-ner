from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"
DATASET_NAME = "AliFartout/PEYMA-ARMAN-Mixed"
IGNORE_INDEX = -100


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(sample):
    encoding = tokenizer(
        sample["tokens"],
        is_split_into_words=True,
        truncation=True,
    )

    word_ids = encoding.word_ids()

    labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            labels.append(IGNORE_INDEX)

        elif word_id != previous_word_id:
            labels.append(sample["ner_tags"][word_id])

        else:
            labels.append(IGNORE_INDEX)

        previous_word_id = word_id

    encoding["labels"] = labels

    return encoding


def preprocess_dataset(dataset):
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset


if __name__ == "__main__":
    dataset = load_dataset(DATASET_NAME)
    tokenized_dataset = preprocess_dataset(dataset)

    print(tokenized_dataset)

    print("\nFirst tokenized sample:")
    print(tokenized_dataset["train"][0])

    for split in tokenized_dataset:
        for sample in tokenized_dataset[split]:
            assert len(sample["input_ids"]) == len(sample["attention_mask"])
            assert len(sample["input_ids"]) == len(sample["labels"])

    print("\nAll alignment checks passed.")