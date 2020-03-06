import pandas as pd
from text_preparation import tokenize

df = pd.read_csv("./data/prepared_dataset.csv", index_col="Id")
df = df.iloc[:10]
df = tokenize(df)
