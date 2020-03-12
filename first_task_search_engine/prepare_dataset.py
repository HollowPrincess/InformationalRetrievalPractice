import numpy as np
import pandas as pd
from polyglot.detect import Detector


def get_lang_info(text: str):
    try:
        lang = Detector(text, quiet=True).language
    except Exception:
        lang = np.nan
    return lang


def run_prepare_dataset():
    df = pd.read_csv(
        "./data/Questions.csv", encoding="iso-8859-1", usecols=["Title", "Body"]
    )

    # concat title and body in one field
    df["Text"] = df["Title"] + " " + df["Body"]
    df.drop(columns=["Title", "Body"], inplace=True)

    # get info about language
    df["lang_info"] = df["Text"].apply(lambda text: get_lang_info(text))
    df.dropna(inplace=True)
    df["lang"] = df["lang_info"].apply(lambda x: (x.name, x.confidence))
    df["lang"], df["confidence"] = zip(*df.lang)

    df.query('confidence > 90 and lang == "английский"', inplace=True)
    df.drop(columns=["lang_info", "lang", "confidence"], inplace=True)
    df.to_csv("./data/prepared_dataset.csv")


if __name__ == "__main__":
    run_prepare_dataset()
