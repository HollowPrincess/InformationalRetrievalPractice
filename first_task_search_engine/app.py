import os
from query_processing import return_documents
from prepare_embeddings import prepare_embeddings_for_dataset
from deeppavlov import build_model, configs

CONFIG_PATH = configs.spelling_correction.brillmoore_wikitypos_en


def run_app():
    model = build_model(CONFIG_PATH, download=True)
    print()
    print("This is a boolean search model.")
    print("For search you need to write a query in format:")
    print()
    print(
        """  first_word first_operator second_word \
        second_operator third_word etc..."""
    )
    print()
    print("where operators can be: and, or, andnot, ornot")
    print("Сase is not important.")
    print()
    print("Please, write your query:")
    query: str = input()
    propose_query: str = model([query])[0]
    print()
    if query != propose_query:
        print('Did you mean "' + propose_query + '"?')
        print("Print yes or no:")
        answer: str = input()
        if answer[0] == "y":
            query = propose_query
        else:
            new_propose_query = query.split(" ")
            for word_pos in range(0, len(new_propose_query), 2):
                new_propose_query[word_pos] = model(
                    [new_propose_query[word_pos]]
                )[0]
            new_propose_query = " ".join(new_propose_query)
            if (propose_query != new_propose_query) and (
                query != new_propose_query
            ):
                print('Did you mean "' + new_propose_query + '"?')
                print("Print yes or no:")
                answer: str = input()
                if answer[0] == "y":
                    query = new_propose_query
    print()
    return_documents(query)


if __name__ == "__main__":
    # if there are not documents embeddings in the data folder
    # then get them for dataset before running application

    if not os.access("data/embeddings.csv", os.R_OK):
        prepare_embeddings_for_dataset()
    run_app()
