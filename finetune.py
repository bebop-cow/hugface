import zipfile, pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# Load FinBERT with a classification head sized for 3 labels
model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert",
    num_labels=3,
)



# (keep the hf_hub_download line from before — zip_path is already set)
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
            line = raw_line.decode("latin-1").strip()   # note: latin-1, see below
            if not line:
                continue
            text, label = line.rsplit("@",1)                          # blank: split into the two parts
            sentences.append(text)
            labels.append(label)

df = pd.DataFrame({"sentence": sentences, "label": labels})
# Map string labels → integer IDs matching FinBERT's id2label
label2id = {"positive": 0, "negative": 1, "neutral": 2}   # blank: neutral's id

df["labels"] = df["label"].map(label2id)

train_df, test_df = train_test_split(
	df, 
	test_size=0.2,
	stratify = df["labels"],
	random_state = 42,
)

# convert to HF Datasets FIRST
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
test_ds  = Dataset.from_pandas(test_df,  preserve_index=False)

def tokenize_batch(batch):
    return tokenizer(batch["sentence"], padding=True, truncation=True)

# now .map() is the HF version, which accepts batched=True
train_tok = train_ds.map(tokenize_batch, batched=True)
test_tok  = test_ds.map(tokenize_batch,  batched=True)

print(train_tok)
print(train_tok[0].keys())

# ---- class weights: inverse to frequency ----
# counts in FinBERT-id order: [positive(0), negative(1), neutral(2)]
counts = np.array([887, 420, 2146])
weights = counts.sum() / (len(counts) * counts)

class_weights = torch.tensor(weights, dtype=torch.float)
print(class_weights)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)        # blank: which axis picks the winning class?
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# print(train_df["labels"].value_counts(normalize = True))
# print(test_df["labels"].value_counts(normalize = True))

# #converts both to HF datasets for the Trainer
# trains_ds = Dataset.from_pandas(train_df, preserve_index=False)
# test_ds = Dataset.from_pandas(test_df, preserve_index=False)

# print(df[["label", "labels"]].head())
# print(df["labels"].value_counts()) 

# print(df.shape)
# print(df.head())
# print(df["label"].value_counts())