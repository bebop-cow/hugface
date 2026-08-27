from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Create a sentiment-analysis pipeline using FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", output_attentions =True)


headlines = [
    "Company posts record profit, raises guidance",
    "Firm slashes dividend amid mounting losses",
    "Board to meet Thursday to review options",
]
inputs = tokenizer(headline, return_tensors="pt", padding=True, truncations=True)

with torch.no_grad():
	outputs = model(**inputs)

# outputs.logits is shape (num_headlines, 3) — raw scores, not yet probabilities
logits = outputs.logits

# Turn each row of logits into a probability distribution over the 3 classes
probs = F.softmax(logits, dim=1)

print(probs)
print(model.config.id2label)            # tells us which column is which class