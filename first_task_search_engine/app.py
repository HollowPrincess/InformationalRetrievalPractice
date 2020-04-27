import os
from query_processing import return_documents
from prepare_embeddings import prepare_embeddings_for_dataset


def run_app():
    print("This is a boolean search model.")
    print("For search you need to write a query in format:")
    print()
    print(
        "    first_word first_operator second_word \
        second_operator third_word etc..."
    )
    print()
    print("where operators can be: and, or, andnot, ornot")
    print("Сase is not important.")
    print()
    print("Please, write your query:")
    query: str = input()
    print()
    return_documents(query)


if __name__ == "__main__":
    data_files = os.listdir("data")
    if "embeddings.csv" not in data_files:
        prepare_embeddings_for_dataset()
    run_app()
