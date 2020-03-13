from query_processing import return_documents


def run_app():
    print("This is a boolean search model.")
    print("For search you need to write a query in format:")
    print()
    print(
        "    first_word first_operator second_word \
        second_operator third_word etc..."
    )
    print()
    print("where operators can be: and, or, not, &, |, ~")
    print("Сase is not important.")
    print()
    print("Please, write your query:")
    query: str = input()
    print()
    return_documents(query)


if __name__ == "__main__":
    run_app()
