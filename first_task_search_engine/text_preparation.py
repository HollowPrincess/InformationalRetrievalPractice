import pandas as pd
from nltk.stem import WordNetLemmatizer


def tokenize(df: pd.DataFrame(columns=["Text"])):
    df["Text"].replace(r"(<[^>]*>|\W)", " ", regex=True, inplace=True)
    df["Text"].replace(r" +", " ", regex=True, inplace=True)
    df["Text"] = df["Text"].astype(str).str.lower()
    df["Text"] = df["Text"].apply(lambda text: text.split(" "))
    return df


def lemmatize(df: pd.DataFrame(columns=["Text"])):
    lemm = WordNetLemmatizer()
    df["Text"] = df["Text"].apply(
        lambda tokens_list: [lemm.lemmatize(w) for w in tokens_list if w != ""]
    )
    return df
