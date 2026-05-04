import pandas as pd

df = pd.read_csv('gemini_1_token_errors.csv')

df = df[df['word'] == 'saline']

print(df.head(20))