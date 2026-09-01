import re
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
from pipeline import score_headlines
import pandas as pd

text = open("speech.txt").read()

# remove any (timestamp) = parens containing digits and colons
text = re.sub(r"\(\d[\d:]*\)", "", text)

# remove speaker labels like "President Trump:" at line starts
text = re.sub(r"^[A-Z][a-zA-Z .]+:", "", text, flags=re.MULTILINE)

# collapse the blank lines left behind
text = re.sub(r"\n\s*\n", "\n", text).strip()

sentences = sent_tokenize(text)
print(len(sentences))

#score in chunks of 32 to keep memory sane
results = []
for i in range(0, len(sentences), 32):
	chunk = sentences[i:i+32]
	results.extend(score_headlines(chunk))

 
print(results[0])

# pair each sentence with its signed score
scored = list(zip(sentences, [r["signed"] for r in results]))

# sort by signed score
scored.sort(key=lambda x: x[1])

print("MOST NEGATIVE:")
for s, sig in scored[:5]:
	print(round(sig,3), s)

print("\n MOST POSITIVE:")
for s, sig in scored[-5:]:
	print(round(sig,3), s)

cols = ["date", "time", "open", "high", "low", "close", "volume"]
df = pd.read_csv("ibm_1min.csv", names=cols)

print(df.shape)
print(df.head())
print(df.tail())


