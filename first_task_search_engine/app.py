from query_processing import return_documents

print("This is a boolean search model.")
print("For search you need to write a query in format:")
print()
print("    first_word first_operator second_word second_operator third_word")
print()
print("where operators can be: and, or, not, &, |, ~")
print("Сase is not important.")
print()
query = input()
print()
return_documents(query)
