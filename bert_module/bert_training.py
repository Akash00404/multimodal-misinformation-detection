import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -----------------------------
# Load datasets
# -----------------------------
fake_df = pd.read_csv("/Users/akash/Desktop/multimodal_news/data/raw/Fake.csv")
real_df = pd.read_csv("/Users/akash/Desktop/multimodal_news/data/raw/True.csv")

fake_df["label"] = 0
real_df["label"] = 1

fake_df = fake_df[["text", "label"]]
real_df = real_df[["text", "label"]]

df = pd.concat([fake_df, real_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

X = df["text"].values
y = df["label"].values

# -----------------------------
# Train / Validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_texts(texts, max_length=256):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

train_encodings = tokenize_texts(X_train)
val_encodings = tokenize_texts(X_val)

train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# -----------------------------
# Dataset class
# -----------------------------
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -----------------------------
# Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./bert_results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=100,
    save_total_limit=1,
    report_to="none"
)


# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# -----------------------------
# Train & Save
# -----------------------------
if __name__ == "__main__":
    trainer.train()
    results = trainer.evaluate()

    print("\n--- EVALUATION RESULTS ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    model.save_pretrained("./bert_fake_news_model")
    tokenizer.save_pretrained("./bert_fake_news_model")

    print("\nBERT model saved to ./bert_fake_news_model")
