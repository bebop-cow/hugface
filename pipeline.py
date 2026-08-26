from transformers import AutoTokenizer

# Create a sentiment-analysis pipeline using FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

headline = "Q3 profit beats estimates; $AAPL raises guidance"

# See the human-readable pieces the headline breaks into
tokens = tokenizer.tokenize(headline)         
print(tokens)                 