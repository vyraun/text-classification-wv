#!/usr/bin/env python
import pandas as pd
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
import time
from sklearn.preprocessing import normalize
import sys
import csv

if __name__ == '__main__':

	start = time.time()
	# The csv file might contain very huge fields, therefore set the field_size_limit to maximum.
	csv.field_size_limit(sys.maxsize)
	# Read train data.
	train_word_vector = pd.read_csv( 'all_v2.tsv', header=0, delimiter="\t")
	# Use the NLTK tokenizer to split the paragraph into sentences.
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = []
	print "Parsing sentences from training set..."	

	# Loop over each news article.
	for review in train_word_vector["news"]:
		try:
			# Split a review into parsed sentences.
			sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
		except:
			continue

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

	num_features = int(sys.argv[1])     # Word vector dimensionality
	min_word_count = 20   # Minimum word count
	num_workers = 40       # Number of threads to run in parallel
	context = 10          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words

	print "Training Word2Vec model..."
	# Train Word2Vec model.
	model = Word2Vec(sentences, workers=num_workers, hs = 0, sg = 1, negative = 10, iter = 25,\
	            size=num_features, min_count = min_word_count, \
	            window = context, sample = downsampling, seed=1)

	model_name = "trained_newsgroup_{0}.txt".format(num_features)
	model.init_sims(replace=True)
	# Save Word2Vec model.
	print "Saving Word2Vec model..."
	#word_vectors = model.wv
	#word_vectors.save(model_name)
	model.save("temp_model")
	print("Saved the Model")
	model = Word2Vec.load("temp_model")
	
	words = list(model.wv.vocab)
	
	filename_reduced = "embeddings_on_newsgroup_{}.txt".format(num_features)
	embedding_file = open(filename_reduced, 'w')

	for i, word in enumerate(words):
        	embedding_file.write("%s\t" % word)
        	for t in model.wv[word]:
                	embedding_file.write("%f\t" % t)
        
        	embedding_file.write("\n")
		
	# Placing a 'unk' token with zero values
	embedding_file.write("unk\t")
	unk = np.zeros(300, dtype='float32')
	for t in unk:
    		embedding_file.write("%f\t" % t)
	embedding_file.write("\n")
	
	endmodeltime = time.time()

	print "time : ", endmodeltime-start
