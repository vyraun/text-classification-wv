from gensim.models import Word2Vec
import sys

model_name = sys.argv[1]

model = Word2Vec.load(model_name)
# Get wordvectors for all words in vocabulary.
word_vectors = model.wv.syn0

print(model)
print(word_vectors.shape)

words = list(model.wv.vocab)
print(words)

print(model['yellow'])

# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
# model.wv['word']

#sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
#			['this', 'is', 'the', 'second', 'sentence'],
#			['yet', 'another', 'sentence'],
#			['one', 'more', 'sentence'],
#			['and', 'the', 'final', 'sentence']]
# train model
#model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
# print(model)
# summarize vocabulary
# words = list(model.wv.vocab)
# print(words)
# access vector for one word
# print(model['sentence'])
# save model
# model.save('model.bin')
# load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)
