import re
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
from pipeline import score_headlines

#score in chunks of 32 to keep memory sane
results = []
for i in range(0, len(sentences), 32):
	chunk = sentences[i:i+32]
	results.extend(score_headlines(chunk))

print(len(results)) text
print(results[0])

text = open("speech.txt").read()


# remove any (timestamp) = parens containing digits and colons
text = re.sub(r"\(\d[\d:]*\)", "", text)

# remove speaker labels like "President Trump:" at line starts
text = re.sub(r"^[A-Z][a-zA-Z .]+:", "", text, flags=re.MULTILINE)

# collapse the blank lines left behind
text = re.sub(r"\n\s*\n", "\n", text).strip()

sentences = sent_tokenize(text)

print(len(sentences))
print(sentences[:5])

