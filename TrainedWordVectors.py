from gensim.models import Word2Vec
import sys

model_name = sys.argv[1]

model = Word2Vec.load(model_name)
# Get wordvectors for all words in vocabulary.
word_vectors = model.wv.syn0

print(word_vectors)

# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
# model.wv['word']
