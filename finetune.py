import zipfile, pandas as pd
from datasets import Dataset

# (keep the hf_hub_download line from before — zip_path is already set)

inner = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"

sentences, labels = [], []
with zipfile.ZipFile(zip_path) as z:
    with z.open(inner) as f:
        for raw_line in f:
            line = raw_line.decode("latin-1").strip()   # note: latin-1, see below
            if not line:
                continue
            text, label = ????                          # blank: split into the two parts
            sentences.append(text)
            labels.append(label)

df = pd.DataFrame({"sentence": sentences, "label": labels})
print(df.shape)
print(df.head())
print(df["label"].value_counts())