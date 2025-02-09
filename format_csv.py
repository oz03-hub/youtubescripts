import pandas as pd

df = pd.read_csv('englishvids.csv', usecols=["id"], encoding_errors="ignore")

print(df.head())
print(len(df))

df.to_csv("formatted.csv", index=False, header=None)