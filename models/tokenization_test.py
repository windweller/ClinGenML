from util import BasicTokenizer

tokenizer = BasicTokenizer(False)

tokens = tokenizer.tokenize(u"Impaired binding of 14-3-3 to C-RAF in Noonan syndrome suggests new approaches in diseases with increased Ras signaling.")

print(tokens)

tokens = tokenizer.tokenize(u"Allelic variants of the Melanocortin 4 receptor (MC4R) gene in a South African study group.")

print(tokens)