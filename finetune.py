import zipfile, pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split


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
	test_siz=0.2,
	stratify = df["labels"],
	random_state = 42,
)

print(train_df["labels"].value_counts(normalize = True))
print(test_df["labels"].value_counts(normalize = True))

#converts both to HF datasets for the Trainer
trains_ds = Dataset.from_pandas(train_df, preserve_index=False)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)

# print(df[["label", "labels"]].head())
# print(df["labels"].value_counts()) 

# print(df.shape)
# print(df.head())
# print(df["label"].value_counts())