from transformers import pipeline

# Create a sentiment-analysis pipeline using FinBERT
clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")

headline = "Company X posts record quarterly profit, raises full-year guidance"

result = clf(headline)          
print(result)                 