import numpy as np
import pandas as pd
from polyglot.detect import Detector


def get_lang_info(text):
    try:
        lang = Detector(text, quiet=True).language
    except Exception:
        lang = np.nan
    return lang


df = pd.read_csv(
    "./data/Questions.csv",
    encoding="iso-8859-1",
    usecols=["Id", "Title", "Body"],
    index_col="Id",
)

# concat title and body in one field
df["Text"] = df["Title"] + " " + df["Body"]
df = df.drop(columns=["Title", "Body"])

# get info about language
df["lang_info"] = df["Text"].apply(lambda text: get_lang_info(text))
df = df.dropna()
df["lang"] = df["lang_info"].apply(lambda x: (x.name, x.confidence))
df["lang"], df["confidence"] = zip(*df.lang)

df = df.loc[(df.confidence > 90) & (df.lang == "английский")]
df = df.drop(columns=["lang_info", "lang", "confidence"])
df.to_csv("./data/prepared_dataset.csv")
