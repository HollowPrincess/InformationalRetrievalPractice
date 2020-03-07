import pandas as pd
from text_preparation import tokenize, lemmatize


df = pd.read_csv("./data/prepared_dataset.csv")
df = df.iloc[:3]
df = tokenize(df)
df = lemmatize(df)
