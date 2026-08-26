from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Create a sentiment-analysis pipeline using FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", output_attentions =True)


headline = "profit beats estimates"
inputs = tokenizer(headline, return_tensors="pt")

with torch.no_grad():
	outputs = model(**inputs)

#output.attentions is a tuple:one attention tensor per layer (12 of them)
attn = outputs.attentions 		#tuple of 12
first_layer = attn[0]
print(first_layer.shape)

# each layer's shape is (batch, heads, tokens, tokens)
# grab layer 0, head 0, to get one clean (tokens x tokens) grid:
one_head = first_layer[0, 0]
print(one_head)