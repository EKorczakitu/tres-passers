import pandas as pd
# 1. Read the pre-existing error file
df = pd.read_csv("project/biobert_1_token_errors.csv")

# 2. Filter for words with length >= 11 characters
df['word'] = df['word'].astype(str).str.strip()
long_words = df[df['word'].str.len() >= 11]

# 3. Get frequency counts, sort (done automatically by value_counts), and print
print("Top 15 most frequent 11+ char words kept as 1 token that BioBERT missed:")
print("-" * 70)
print(long_words['word'].value_counts().head(15))
print("-" * 70)
