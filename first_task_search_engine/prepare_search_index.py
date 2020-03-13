from collections import defaultdict
from collections import Counter
import functools
import math
import operator
import os
import pandas as pd
import pickle
import sys
import ndjson
from text_preparation import tokenize, lemmatize

const_size_in_bytes: int = 1024 * 1024 * 1


def write_dict_to_file(file_number: int, dict_for_index: defaultdict(list)) -> int:
    """
    This function write parts of index from texts blocks in tmp index file.
    First stage of creating index with SPIMI.
    """
    for key in dict_for_index.keys():
        term_freqs = Counter(dict_for_index[key])  # count terms freqs in docs
        dict_for_index[key] = term_freqs
    dict_for_index = sorted(dict_for_index.items())
    with open("data/index_blocks/index_file{}.txt".format(str(file_number)), "w") as f:
        ndjson.dump(dict_for_index, f)
        # write as json for comfortable reading lines
        file_number += 1  # this var is for naming tmp index files
    return file_number


def SPIMI_invert(block: pd.DataFrame(columns=["Text"]), file_number: int) -> int:
    """
    This function contains SPIMI algo for buildin index.
    """
    dict_for_index = defaultdict(list)
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
    min_term = min(readed_dict.keys())  # get first term in a queue
    term_postings = readed_dict.pop(min_term)  # get term posting list

    if files_dict:
        files_to_read = files_dict.pop(
            min_term
        )  # get files pointers for future reading
    # merging terms postings lists with frequnces:
    term_postings = dict(functools.reduce(operator.add, map(Counter, term_postings)))

    line_to_write = {min_term: term_postings}
    if (
        sys.getsizeof(preparred_part) + sys.getsizeof(line_to_write)
    ) > const_size_in_bytes:
        # dict reaches memory limit
        file_name = "data/separated_index/{}.txt".format(min_term)
        # the first term locates in a next file
        # it will be a name of current file for simplify search
        with open(file_name, "wb") as handle:
            pickle.dump(preparred_part, handle)
        preparred_part = defaultdict(list)

    preparred_part.update(line_to_write)
    return readed_dict, preparred_part, files_to_read, files_dict


def merging_tmp_index(file_number: int):
    # clean dir for index files:
    index_parts = os.listdir("data/separated_index")
    for file in index_parts:
        os.remove("data/separated_index/" + file)
    # open files with tmp index from first stage:
    files = [
        open(
            "data/index_blocks/index_file{}.txt".format(str(i)),
            "r",
            buffering=math.floor(const_size_in_bytes / file_number),
        )
        for i in range(1, file_number)
    ]

    readed_dict = defaultdict(list)  # queue of terms and postings
    preparred_part = defaultdict(list)  # queue of terms for writing
    files_to_read = files.copy()  # pointers on files for loop
    files_dict = defaultdict(list)
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
            ) = prepare_line_for_writing(readed_dict, preparred_part, files_dict)

    # write tails in dicts which caused by not reaching memory limit for writing:
    while len(readed_dict) > 0 | len(preparred_part) > 0:
        if readed_dict:
            (
                readed_dict,
                preparred_part,
                files_to_read,
                files_dict,
            ) = prepare_line_for_writing(readed_dict, preparred_part, files_dict)
        else:
            max_term = max(preparred_part.keys())

            file_name = "data/separated_index/{}.txt".format(max_term)
            with open(file_name, "wb") as handle:
                pickle.dump(preparred_part, handle)
            preparred_part = defaultdict(list)


def run_index_prep():
    df = pd.read_csv("./data/prepared_dataset.csv", index_col=0)
    df = tokenize(df)
    df = lemmatize(df)

    # how many documents prepare in one time:
    block_size: int = 10000
    # var for naming files with tmp index:
    file_number: int = 1

    for block_num in range(int(math.ceil(df.shape[0] / block_size))):
        block = df.iloc[block_num * block_size : (block_num + 1) * block_size]
        file_number = SPIMI_invert(block, file_number)
    merging_tmp_index(file_number)


if __name__ == "__main__":
    run_index_prep()
