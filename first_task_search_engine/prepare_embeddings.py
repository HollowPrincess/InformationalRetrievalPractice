import os
import numpy as np
import pandas as pd
from deeppavlov.core.data.utils import download
from deeppavlov.models.embedders.glove_embedder import GloVeEmbedder
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from nptyping import Array


def cosine_similarity(x: Array[float], y: Array[float]) -> float:
    """Return the cosine similarity of two vectors."""
    numerator: float = sum(np.multiply(x, y))
    denominator: float = np.sqrt(
        (sum(np.multiply(x, x))) * (sum(np.multiply(y, y)))
    )
    return numerator / denominator


def prepare_embeddings_for_query(query: str) -> Array[float]:
    """Return an embedding for query string."""
    tokenizer = NLTKTokenizer()
    embedder = GloVeEmbedder(load_path="data/glove.6B.100d.txt", pad_zero=True)
    embed_query: Array[float] = embedder(tokenizer([query]), mean=True)[0]
    return embed_query


def prepare_embeddings_for_dataset():
    """Write a file with documents embeddings."""
    print("The first run may take several minutes for preparation.")
    data_files = os.listdir("data")
    if "glove.6B.100d.txt" not in data_files:
        download(
            "data/glove.6B.100d.txt",
            source_url="http://files.deeppavlov.ai/embeddings/glove.6B.100d.txt",
        )

    tokenizer = NLTKTokenizer()
    embedder = GloVeEmbedder(load_path="data/glove.6B.100d.txt", pad_zero=True)

    with open("data/embeddings.csv", "w") as emb_file:
        for chunk in pd.read_csv(
            "data/prepared_dataset.csv",
            header=None,
            chunksize=10000,
            names=["Text"],
        ):
            embeddings: Array[float] = embedder(
                tokenizer(chunk.loc[:, "Text"]), mean=True
            )
            np.savetxt(emb_file, embeddings, delimiter=",")


if __name__ == "__main__":
    prepare_embeddings_for_dataset()
