from collections import defaultdict
from collections import Counter
import functools
import math
import operator
import os
import pickle
import sys
import numpy as np
import pandas as pd
import ndjson
from text_preparation import tokenize, lemmatize
from constants import const_size_in_bytes


def write_dict_to_file(
    file_number: int, dict_for_index: defaultdict(list)
) -> int:
    """
    This function write parts of index from texts blocks in tmp index file.
    First stage of creating index with SPIMI.
    """
    for key in dict_for_index.keys():
        term_freqs = Counter(dict_for_index[key])  # count terms freqs in docs
        dict_for_index[key] = term_freqs
    dict_for_index: dict = sorted(dict_for_index.items())
    with open(
        "data/index_blocks/index_file{}.txt".format(str(file_number)), "w"
    ) as f:
        ndjson.dump(dict_for_index, f)
        # write as json for comfortable reading lines
        file_number += 1  # this var is for naming tmp index files
    return file_number


def SPIMI_invert(
    block: pd.DataFrame(columns=["Text"]), file_number: int
) -> int:
    """
    This function contains SPIMI algo for buildin index.
    """
    dict_for_index: defaultdict(list) = defaultdict(list)
    for docID, terms_list in block.iterrows():
        for term in terms_list.values[0]:
            if sys.getsizeof(dict_for_index) < const_size_in_bytes:
                dict_for_index[term].append(docID)
            else:
                # write index of part of block to tmp file
                # before we reach limit of memory
                file_number = write_dict_to_file(file_number, dict_for_index)
                dict_for_index = defaultdict(list)
    if dict_for_index:
        # writing a tail of current index part
        file_number = write_dict_to_file(file_number, dict_for_index)
    return file_number


def prepare_line_for_writing(
    readed_dict: defaultdict(list),
    preparred_part: defaultdict(list),
    files_dict: defaultdict(list),
) -> [defaultdict(list), defaultdict(list), list, defaultdict(list)]:
    """
    Writing lines in files with fully-prepared index.
    """
    min_term: str = min(readed_dict.keys())  # get first term in a queue
    term_postings: list = readed_dict.pop(min_term)  # get term posting list

    if files_dict:
        files_to_read = files_dict.pop(
            min_term
        )  # get files pointers for future reading
    # merging terms postings lists with frequnces:
    term_postings: dict = dict(
        functools.reduce(operator.add, map(Counter, term_postings))
    )

    line_to_write: dict = {min_term: term_postings}
    if (
        sys.getsizeof(preparred_part) + sys.getsizeof(line_to_write)
    ) > const_size_in_bytes:
        # dict reaches memory limit
        file_name: str = "data/separated_index/{}.txt".format(min_term)
        # the first term locates in a next file
        # it will be a name of current file for simplify search
        with open(file_name, "wb") as handle:
            pickle.dump(preparred_part, handle)
        preparred_part = defaultdict(list)

    preparred_part.update(line_to_write)
    return readed_dict, preparred_part, files_to_read, files_dict


def merging_tmp_index(file_number: int):
    # clean dir for index files:
    index_parts: list = os.listdir("data/separated_index")
    for file in index_parts:
        os.remove("data/separated_index/" + file)
    # open files with tmp index from first stage:
    files: list = [
        open(
            "data/index_blocks/index_file{}.txt".format(str(i)),
            "r",
            buffering=math.floor(const_size_in_bytes / file_number),
        )
        for i in range(1, file_number)
    ]

    readed_dict: defaultdict(list) = defaultdict(
        list
    )  # queue of terms and postings
    preparred_part: defaultdict(list) = defaultdict(
        list
    )  # queue of terms for writing
    files_to_read: list = files.copy()  # pointers on files for loop
    files_dict: defaultdict(list) = defaultdict(list)
    # pointers on files which contains terms from keys
    while files:  # while reading any file
        for file in files_to_read:
            if not file.closed:
                # get new term from file if it is in reading mode:
                data = file.readline()
                if data:
                    k, v = ndjson.loads(data)[0]
                    readed_dict[k].append(v)
                    files_dict[k].append(file)
                else:
                    files.remove(file)
                    file.close()
                    if os.path.exists(file.name):
                        os.remove(file.name)  # drop tmp index file

        if readed_dict:
            (
                readed_dict,
                preparred_part,
                files_to_read,
                files_dict,
            ) = prepare_line_for_writing(
                readed_dict, preparred_part, files_dict
            )

    # write tails in dicts which caused by
    # not reaching memory limit for writing:
    while len(readed_dict) > 0 | len(preparred_part) > 0:
        if readed_dict:
            (
                readed_dict,
                preparred_part,
                files_to_read,
                files_dict,
            ) = prepare_line_for_writing(
                readed_dict, preparred_part, files_dict
            )
        else:
            max_term = max(preparred_part.keys())

            file_name = "data/separated_index/{}.txt".format(max_term)
            with open(file_name, "wb") as handle:
                pickle.dump(preparred_part, handle)
            preparred_part = defaultdict(list)


def run_index_prep():
    # var for naming files with tmp index:
    file_number: int = 1

    for chunk in pd.read_csv(
        "data/prepared_dataset.csv",
        header=None,
        chunksize=10000,
        names=["Text"],
    ):
        chunk: pd.DataFrame(columns=["Text"]) = tokenize(chunk)
        chunk = lemmatize(chunk)
        file_number = SPIMI_invert(chunk, file_number)
    max_doc_id: int = np.max(list(chunk.index))
    merging_tmp_index(file_number)
    return max_doc_id


if __name__ == "__main__":
    for folder in ["data/index_blocks", "data/separated_index"]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    max_doc_id = run_index_prep()

    with open("constants.py", "r") as fin, open(
        "constants_tmp.py", "w"
    ) as fout:
        for line in fin:
            if line.find("max_doc_id") == -1:
                fout.write(line)
            else:
                fout.write("max_doc_id:int = {}\n".format(max_doc_id))

    if os.path.exists("constants.py"):
        os.remove("constants.py")
    if os.path.exists("constants_tmp.py"):
        os.rename("constants_tmp.py", "constants.py")
