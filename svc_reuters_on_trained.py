import time
from gensim.models import Word2Vec
import pandas as pd
import time
from nltk.corpus import stopwords
import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility
from numpy import float32
import math
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn import svm
import pickle
import cPickle
from math import *
from sklearn.mixture import GaussianMixture
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame
from gensim.models import KeyedVectors

filename = sys.argv[1]
dims = int(sys.argv[2])

# Word Vectors
Glove = KeyedVectors.load(filename)

start = time.time()

all = pd.read_pickle('all.pkl')

# Computing tf-idf values.
traindata = []
for i in range( 0, len(all["text"])):
	traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["text"][i], True)))

tfv = TfidfVectorizer(strip_accents='unicode',dtype=np.float32)
tfidfmatrix_traindata = tfv.fit_transform(traindata)
featurenames = tfv.get_feature_names()
idf = tfv._tfidf.idf_

# Creating a dictionary with word mapped to its idf value 
print "Creating word-idf dictionary for Training set..."

word_idf_dict = {}
for pair in zip(featurenames, idf):
	word_idf_dict[pair[0]] = pair[1]

temp_time = time.time() - start
print "Creating Document Vectors...:", temp_time, "seconds."

# Create train and text data.
lb = MultiLabelBinarizer()
Y = lb.fit_transform(all.tags)
train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

train = DataFrame({'text': []})
test = DataFrame({'text': []})

train["text"] = train_data.reset_index(drop=True)
test["text"] = test_data.reset_index(drop=True)

# gwbowv is a matrix which contain normalised normalised gwbowv.
gwbowv = np.zeros( (train["text"].size, dims), dtype="float32")

counter = 0

min_no = 0
max_no = 0
for review in train["text"]:
	# Get the wordlist in each text article.
	words = KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True )
	
	word_to_tf = {}

	for x in words:
	    if x in word_to_tf:
	        word_to_tf[x] += 1
	    else:
		word_to_tf[x] = 1
		
	document_vector = np.zeros(dims, dtype="float32")
	
	for word in word_to_tf:
		if word in Glove:
			document_vector += (word_to_tf[word]*word_idf_dict[word]) * Glove[word]
			#print("Word Available")
		else:
			pass
			#print("X X X Word Not Available")
		#if word in Glove:
		#document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove[word]
		#else:
		#    document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove['unk']
		    	
	# These words will be passed to generate the document vector
	gwbowv[counter] = document_vector # Code to construct the document vector
	
	counter+=1
	if counter % 1000 == 0:
		print "Train text Covered : ",counter
		#print "Document Vector", document_vector


endtime_gwbowv = time.time() - start

gwbowv_test = np.zeros( (test["text"].size, dims), dtype="float32")

counter = 0

for review in test["text"]:
	# Get the wordlist in each text article.
	words = KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True )
	
	word_to_tf = {}

	for x in words:
	    if x in word_to_tf:
	        word_to_tf[x] += 1
	    else:
		word_to_tf[x] = 1
		
	document_vector = np.zeros(dims, dtype="float32")
	
	for word in word_to_tf:
		if word in Glove:
			document_vector += (word_to_tf[word]*word_idf_dict[word]) * Glove[word]
			#print("Word Available")
		else:
			pass
			#print("X X X Word Not Available")
			
		#if word in Glove:
		#document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove[word]
		#else:
		#    document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove['unk']
		    	
	# These words will be passed to generate the document vector
	gwbowv[counter] = document_vector # Code to construct the document vector
	
	counter+=1
	if counter % 1000 == 0:
		print "Test Text Covered : ",counter
		#print "Document Vector", document_vector

#saving gwbowv train and test matrices
np.save('Reuters_train_trained_{}'.format(dims), gwbowv)
np.save('Reuters_test_trained_{}'.format(dims), gwbowv_test)

endtime = time.time() - start
print "Total time taken: ", endtime, "seconds." 

print "********************************************************"
