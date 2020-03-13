import numpy as np
import pandas as pd
from polyglot.detect import Detector
from polyglot.detect.base import Language
from polyglot.detect.base import UnknownLanguage
from typing import Union
from pycld2 import error


def get_lang_info(text: str) -> Union[Language, "float"]:
    """
    File contains questions in different languages.
    We get only English texts.
    """
    try:
        lang = Detector(text, quiet=True).language
    except (error, UnknownLanguage):
        lang = np.nan
    return lang


def run_prepare_dataset(
    df: pd.DataFrame(columns=["Title", "Body"])
) -> pd.DataFrame(columns=["Text"]):
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

    return df


if __name__ == "__main__":
    with open("data/prepared_dataset.csv", "w") as prep_file:
        for chunk, counter in zip(
            pd.read_csv(
                "data/Questions.csv",
                encoding="iso-8859-1",
                usecols=["Title", "Body"],
                chunksize=10000,
            ),
            range(20),
        ):
            chunk: pd.DataFrame(columns=["Text"]) = run_prepare_dataset(chunk)
            chunk.to_csv(prep_file, header=None)
            if counter == 19:
                break
