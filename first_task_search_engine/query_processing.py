from boolean_search_model import (
    intersect_many_postings_lists,
    union_postings_lists,
    subtract_postings_lists,
    get_negation_from_postings,
)
from prepare_embeddings import (
    prepare_embeddings_for_query,
    cosine_similarity,
)
from collections import deque
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import pandas as pd
import pickle
import re
from typing import Union
from nptyping import Array
from constants import (
    const_size_in_bytes,
    TOP_DOCS_NUMBER,
    LINE_OFFSET_BEFORE_WORD,
)

operations_set: set = set(["and", "or", "ornot", "andnot"])
files: list = sorted(os.listdir("data/separated_index"))
index_slices_words: list = [word[:-4] for word in files]


def get_postings(word: str) -> dict:
    """Returns postings list for a word."""
    postings: dict = {}
    for index_word in index_slices_words:
        if word < index_word:
            file_name: str = "data/separated_index/{}.txt".format(index_word)
            with open(
                file_name, "rb", buffering=const_size_in_bytes
            ) as index_file:
                data: dict = pickle.load(index_file)
                if word in data:
                    postings = data[word]
            break
    if postings:
        postings = dict(sorted(postings.items()))
    return postings


def get_postings_with_query(
    query_words: deque, query_operations: deque
) -> dict:
    """Returns postings list which is a result of query."""

    multiple_and_postings: list = []
    empty_postings_in_multi_end: bool = False

    word_left: str = query_words.popleft()  # first word in query
    lemm = WordNetLemmatizer()
    word_left = lemm.lemmatize(word_left)
    postings_left: dict = get_postings(word_left)
    while True:
        try:
            oper: str = query_operations.popleft()
            word_right: str = query_words.popleft()
            word_right = lemm.lemmatize(word_right)
            postings_right: dict = get_postings(word_right)
            if oper != "and":
                if multiple_and_postings:
                    postings_left = intersect_many_postings_lists(
                        multiple_and_postings
                    )
                    multiple_and_postings = []

                if oper == "or":
                    if postings_right:
                        postings_left = union_postings_lists(
                            postings_left, postings_right
                        )
                elif oper == "andnot":
                    if postings_right:
                        postings_left = subtract_postings_lists(
                            postings_left, postings_right
                        )
                elif oper == "ornot":
                    postings_left = union_postings_lists(
                        postings_left,
                        get_negation_from_postings(postings_right),
                    )
            else:
                if not empty_postings_in_multi_end:
                    if multiple_and_postings:
                        multiple_and_postings.append(postings_right)
                    else:
                        multiple_and_postings = [postings_left, postings_right]
                if not (postings_right or postings_left):
                    empty_postings_in_multi_end = True
                    multiple_and_postings = []
        except IndexError:  # end of a query
            if multiple_and_postings:
                postings_left = intersect_many_postings_lists(
                    multiple_and_postings
                )
                multiple_and_postings = []
            break
    return postings_left


def return_documents(query: str):
    """Returns top documents for a query."""
    query = query.strip().lower()
    query = re.sub(r"\W", " ", query)
    query = re.sub(r" +", " ", query)
    # for vector model we need to take documents which contains all words
    # so we replace and-query by or-query:
    query = re.sub(r" and ", " or ", query)
    query = query.split(" ")

    # prepare query for makng embedding: replace not words and key words:
    not_indexes: Array[int] = np.array(
        [query.index(x) for x in query if x.find("not") != -1]
    )
    not_indexes = np.append(not_indexes, not_indexes + 1)
    query_for_embed: Union[Array[str], str] = np.array(query)
    if len(not_indexes) > 0:
        query_for_embed = np.delete(query_for_embed, not_indexes)
    query_for_embed = query_for_embed[::2]
    query_for_embed = " ".join(query_for_embed)
    # get embedding of importance words in query:
    query_embed: Array[float] = prepare_embeddings_for_query(query_for_embed)

    query_words: deque = deque(query[::2])
    query_operations: deque = deque(query[1::2])
    query_operations_set: set = set(np.unique(query_operations))

    if (
        query_operations_set.intersection(operations_set)
        != query_operations_set
    ):
        print("The query don't match the format.")
        return {}

    postings = {}
    if query:
        postings: dict = get_postings_with_query(
            query_words.copy(), query_operations
        )

    if postings:
        docs_dict: dict = dict(
            sorted(postings.items(), key=lambda x: x[1], reverse=True)
        )
        # sort documents by rank
        docs_ids = np.array(list(docs_dict.keys()), dtype=int)
        print("Documents num:", len(docs_ids))

        ranks = np.array([], dtype=float)
        for docID in docs_ids:
            elem_emb: Array[float] = pd.read_csv(
                "data/embeddings.csv", skiprows=docID, nrows=1, header=None,
            ).values[0]
            ranks = np.append(ranks, cosine_similarity(elem_emb, query_embed))
        ranks = pd.DataFrame({"id": docs_ids, "rank": ranks})
        ranks = ranks.sort_values(by="rank", ascending=False).iloc[
            :TOP_DOCS_NUMBER
        ]

        docs_ids: Array[int] = ranks.loc[:, "id"].values
        docs_ranks: Array[float] = ranks.loc[:, "rank"].values

        print("Top {} documents: \n".format(TOP_DOCS_NUMBER))
        for docID, rank in zip(docs_ids, docs_ranks):
            elem = pd.read_csv(
                "data/prepared_dataset.csv",
                skiprows=docID,
                nrows=1,
                header=None,
            ).values[0]
            print("Document id: ", docID)
            print("Document rank: ", rank, "\n")

            for word in query_words:
                word_pos: int = elem[0].lower().find(word)
                if word_pos != -1:
                    left_pos: int = max(0, word_pos - LINE_OFFSET_BEFORE_WORD)
                    right_pos: int = min(
                        word_pos + LINE_OFFSET_BEFORE_WORD + 1, len(elem[0])
                    )
                    part: str = elem[0][left_pos:right_pos]
                    print(word, "is in:")
                    print(part)
                else:
                    print("This document doesn't contain ", word)
            print()

    else:
        print(
            "There are no documents matching this query\
            in questions collection."
        )
