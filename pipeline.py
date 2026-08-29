from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

# from datasets import load_dataset
# ds = load_dataset("takala/financial_phrasebank", "sentences_75agree")
# print(ds["train"][0])
# print(ds["train"].features)

# Create a sentiment-analysis pipeline using FinBERT
_MODEL = "./finbert-finetuned/checkpoint-519"
# _MODEL = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(_MODEL)
model.eval() #inference mode 


def score_headlines(headlines, min_conf=0.5):
	"""Takes a list of headlines, returns a list of signal dicts"""
	inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)

	with torch.no_grad():
		logits = model(**inputs).logits

	probs = F.softmax(logits, dim = 1)

	results = []
	for row in probs:
		pos, neg, neu = row[0].item(), row[1].item(), row[2].item()
		signed = pos - neg
		conf = max(pos, neg, neu)
		# apply the confidence floor
		if conf < min_conf:
			label = "neutral"
		else:
			label = model.config.id2label[row.argmax().item()]
		results.append({
			"signed": round(signed, 4),
			"label": label, 
			"pos": round(pos, 4), "neg": round(neg, 4), "neu": round(neu, 4),
			})
	return results

# quick test
for r in score_headlines([
    "Company posts record profit, raises guidance",
    "Firm slashes dividend amid mounting losses",
    "Board to meet Thursday to review options",
]):
    print(r)
    print(model.config.id2label)

