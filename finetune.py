"""
finetune.py — Fine-tune FinBERT on Financial PhraseBank for headline sentiment.

Pipeline: load data -> parse -> map labels -> stratified split -> tokenize
          -> class weights -> metrics -> baseline eval -> (train next).

Run once to produce a fine-tuned model on disk; the screener tool loads it later.
"""

import zipfile
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "ProsusAI/finbert"

# ---------------------------------------------------------------------------
# 1. Get the data — download the zip, read the 75%-agreement text file inside.
#    Each line looks like:  some financial sentence@label
# ---------------------------------------------------------------------------
zip_path = hf_hub_download(
    repo_id="takala/financial_phrasebank",
    repo_type="dataset",
    filename="data/FinancialPhraseBank-v1.0.zip",
)

inner = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"

sentences, labels = [], []
with zipfile.ZipFile(zip_path) as z:
    with z.open(inner) as f:
        for raw_line in f:
            line = raw_line.decode("latin-1").strip()  # file isn't UTF-8
            if not line:
                continue
            text, label = line.rsplit("@", 1)          # split on the LAST @
            sentences.append(text)
            labels.append(label)

df = pd.DataFrame({"sentence": sentences, "label": labels})

# ---------------------------------------------------------------------------
# 2. Map string labels -> integer IDs, MATCHING FinBERT's own ordering.
#    (positive=0, negative=1, neutral=2) — must match the pretrained head.
# ---------------------------------------------------------------------------
label2id = {"positive": 0, "negative": 1, "neutral": 2}
df["labels"] = df["label"].map(label2id)

print("Class counts:\n", df["label"].value_counts(), "\n")

# ---------------------------------------------------------------------------
# 3. Stratified train/test split — preserves the 62/26/12 class balance in
#    both halves so the held-out test set is a faithful exam.
# ---------------------------------------------------------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["labels"],
    random_state=42,
)

# ---------------------------------------------------------------------------
# 4. Convert to HF Datasets, then tokenize. Fixed-length padding (=128) keeps
#    every row the same length so the Trainer's collator can batch cleanly.
# ---------------------------------------------------------------------------
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


train_tok = train_ds.map(tokenize_batch, batched=True)
test_tok = test_ds.map(tokenize_batch, batched=True)

# Keep only the columns the model/collator needs (drop the string columns that
# would otherwise confuse the collator).
keep = ["input_ids", "attention_mask", "labels"]
train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in keep])
test_tok = test_tok.remove_columns([c for c in test_tok.column_names if c not in keep])

print("Tokenized columns:", train_tok.column_names, "\n")

# ---------------------------------------------------------------------------
# 5. Class weights — inverse to frequency, so the rare NEGATIVE class gets a
#    louder voice in the loss. Order matches FinBERT ids: [pos, neg, neu].
# ---------------------------------------------------------------------------
counts = np.array([
    (df["labels"] == 0).sum(),  # positive
    (df["labels"] == 1).sum(),  # negative
    (df["labels"] == 2).sum(),  # neutral
])
weights = counts.sum() / (len(counts) * counts)
class_weights = torch.tensor(weights, dtype=torch.float)
print("Class weights [pos, neg, neu]:", class_weights, "\n")

# ---------------------------------------------------------------------------
# 6. Metrics — accuracy as a sanity reference, macro-F1 as the honest number
#    (weights each class equally, so the rare classes actually count).
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# ---------------------------------------------------------------------------
# 7. Load the trainable model and measure the BASELINE on the held-out set
#    BEFORE any fine-tuning — this is the number training must beat.
# ---------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

eval_args = TrainingArguments(
    output_dir="./finbert-finetuned",
    per_device_eval_batch_size=32,
    report_to="none",
)

baseline_trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=test_tok,
    compute_metrics=compute_metrics,
)

print("Baseline (no fine-tuning):")
print(baseline_trainer.evaluate())

# ---------------------------------------------------------------------------
# 8. TRAINING goes here next — a weighted Trainer using class_weights,
#    then re-evaluate and compare macro-F1 against the baseline above.
# ---------------------------------------------------------------------------

import torch.nn as nn

# Custom Trainer: same as the stock one, but loss is class-weighted.
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # weighted cross-entropy — weights live on the same device as the model
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


train_args = TrainingArguments(
    output_dir="./finbert-finetuned",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,               
    eval_strategy="epoch",               # evaluate after each epoch
    save_strategy="epoch",
    learning_rate=2e-5,                  # standard fine-tuning LR for BERT
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=train_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    compute_metrics=compute_metrics,
)

trainer.train()

print("After fine-tuning:")
print(trainer.evaluate())