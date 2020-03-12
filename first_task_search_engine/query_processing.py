import boolean_search_model
from collections import deque
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import pandas as pd
import pickle
import re

operations_set = set(["and", "or", "not", "&", "|", "~"])
const_size_in_bytes = 1024 * 1024 * 100  # type:int #memory limit
files = os.listdir("data/separated_index")  # type:list
index_slices_words = [word[:-4] for word in files]  # type:list


def get_postings(word: str) -> dict:
    """
    This function gets posting list for a word from the index
    """
    postings = {}
    for index_word in index_slices_words:
        if word < index_word:
            file_name = "data/separated_index/{}.txt".format(index_word)
            with open(file_name, "rb", buffering=const_size_in_bytes) as index_file:
                data = pickle.load(index_file)
                if word in data.keys():
                    postings = data[word]
            break
    return postings


def get_postings_with_query(query: str) -> dict:
    """
    This function gets posting list which is a result of operations with words.
    """
    lemm = WordNetLemmatizer()
    query = re.sub(r"\W", " ", query)
    query = re.sub(r" +", " ", query)
    query = query.lower().split(" ")
    query = [lemm.lemmatize(w) for w in query if w != ""]
    query_words = deque(query[::2])
    query_operations = deque(query[1::2])
    query_operations_set = set(np.unique(query_operations))
    if query_operations_set.intersection(operations_set) != query_operations_set:
        print("The query don't match the format.")
        return {}

    multiple_and_postings = []  # type:list
    empty_postings_in_multi_end = False  # type:bool
    if query:
        word_left = query_words.popleft()  # type:str #first word in query
        postings_left = get_postings(word_left)
        while True:
            try:
                oper = query_operations.popleft()  # type:str
                word_right = query_words.popleft()  # type:str
                postings_right = get_postings(word_right)
                if oper in ["or", "|"]:
                    if multiple_and_postings:
                        postings_left = boolean_search_model.intersect_many_postings_lists(
                            multiple_and_postings
                        )
                        multiple_and_postings = []
                    if postings_right:
                        postings_left = boolean_search_model.union_postings_lists(
                            postings_left, postings_right
                        )
                elif oper in ["not", "~"]:
                    if multiple_and_postings:
                        postings_left = boolean_search_model.intersect_many_postings_lists(
                            postings_left, postings_right
                        )
                        multiple_and_postings = []
                    if postings_right:
                        postings_left = boolean_search_model.subtract_postings_lists(
                            postings_left, postings_right
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
                    postings_left = boolean_search_model.intersect_many_postings_lists(
                        multiple_and_postings
                    )
                    multiple_and_postings = []
                break
    return postings_left


def return_documents(query: str):
    query = query.strip()
    postings = get_postings_with_query(query)  # type:dict
    if postings:
        docs_dict = OrderedDict(
            sorted(postings.items(), key=lambda x: x[1], reverse=True)
        )  # sort documents by rank
        docs_ids = np.array(list(docs_dict.keys()), dtype=int)[:5]
        docs_ranks = np.array(list(docs_dict.values()), dtype=int)[:5]
        docs = pd.read_csv("data/prepared_dataset.csv", index_col=0)
        res = docs.iloc[docs_ids].values
        for elem, docID, rank in zip(res, docs_ids, docs_ranks):
            print("Top 5 documents\n")
            print("Document id: ", docID)
            print("Document rank: ", rank, "\n")
            print(elem[0])
    else:
        print("There are no documents matching this query in questions collection.")
