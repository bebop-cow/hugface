import re
import ntlk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

text = open("speech.txt").read()
sentences = sent_tokenize(text)

# remove any (timestamp) = parens containing digits and colons
text = re.sub(r"\(\d[\d:]*\)", "", text)

# remove speaker labels like "President Trump:" at line starts
text = re.sub(r"^[A-Z][a-zA-Z .]+:", "", text, flags=re.MULTILINE)

# collapse the blank lines left behind
text = re.sub(r"\n\s*\n", "\n", text).strip()

print(len[sentences])
print(sentences[:5])

