import time
import warnings
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
from sklearn.svm import SVC, LinearSVC
import pickle
import cPickle
from math import *
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize

if __name__ == '__main__':

	filename = sys.argv[1]
	dims = int(sys.argv[2])

	# Word Vectors
	Glove = {}
	f = open(filename)
	print("Loading Glove vectors.")
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    Glove[word] = coefs
	f.close()

	# Load train data.
	train = pd.read_csv( 'train_v2.tsv', header=0, delimiter="\t")

	# Load test data.
	test = pd.read_csv( 'test_v2.tsv', header=0, delimiter="\t")
	all = pd.read_csv( 'all_v2.tsv', header=0, delimiter="\t")

	# Computing tf-idf values.
	traindata = []
	for i in range( 0, len(all["news"])):
		traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["news"][i], True)))

	tfv = TfidfVectorizer(strip_accents='unicode',dtype=np.float32)
	tfidfmatrix_traindata = tfv.fit_transform(traindata)
	featurenames = tfv.get_feature_names()
	idf = tfv._tfidf.idf_

	# Creating a dictionary with word mapped to its idf value 
	print "Creating word-idf dictionary for Training set..."

	word_idf_dict = {}
	for pair in zip(featurenames, idf):
		word_idf_dict[pair[0]] = pair[1]

	# gwbowv is a matrix which contains normalised document vectors.
	gwbowv = np.zeros( (train["news"].size, dims), dtype="float32")

	counter = 0

	for review in train["news"]:
		# Get the wordlist in each news article.
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
			    #print("To be added vector: ")
			    #print word_to_tf[word]*word_idf_dict[word] * Glove[word]
			    document_vector += (word_to_tf[word]*word_idf_dict[word]) * Glove[word]  # 
			    #print("Found TF-IDF" + str(word_to_tf[word]) +" "+ str(word_idf_dict[word]))
			else:
			    document_vector += (word_to_tf[word]*word_idf_dict[word]) * Glove['unk'] # word_idf_dict[word]
			    #print("UNK TF-IDF" + str(word_to_tf[word]) +" "+ str(word_idf_dict[word]))

		# These words will be passed to generate the document vector
		gwbowv[counter] = document_vector # Code to construct the document vector
		counter+=1
		if counter % 1000 == 0:
			print "Train News Covered : ",counter
			#print "Document Vector", document_vector.shape

	gwbowv_test = np.zeros( (test["news"].size, dims), dtype="float32")

	counter = 0

	for review in test["news"]:
		# Get the wordlist in each news article.
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
			    document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove[word]
			else:
			    document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove['unk']

		gwbowv_test[counter] = document_vector
		counter+=1
		if counter % 1000 == 0:
			print "Test News Covered : ",counter
			#print "Document Vector", document_vector

	#saving gwbowv train and test matrices
	np.save("NewsGroup_dv_training", gwbowv)
	np.save("Newsgroup_dv_test", gwbowv_test)

	print "Fitting a SVM classifier on labeled training data..."

	print "********************************************************"

	#clf =  LinearSVC(C=1)
	#clf.fit(gwbowv, train["class"])
	#Y_true, Y_pred  = test["class"], clf.predict(gwbowv_test)
	#print "Report"
	#print classification_report(Y_true, Y_pred, digits=6)
	#print "Accuracy: ",clf.score(gwbowv_test,test["class"])
        
	param_grid = [{'C': np.arange(0.1, 7, 0.2)}]
	scores = ['f1_weighted']
	#scores = ['accuracy', 'recall_micro', 'f1_micro' , 'precision_micro', 'recall_macro', 'f1_macro' , 'precision_macro', 'recall_weighted', 'f1_weighted' , 'precision_weighted'] #, 'accuracy', 'recall', 'f1']
	for score in scores:
	    strt = time.time()
	    print "# Tuning hyper-parameters for", score, "\n"
	    clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=5, scoring= '%s' % score)
	    clf.fit(gwbowv, train["class"])
	    print "Best parameters set found on development set:\n"
	    print clf.best_params_
	    print "Best value for ", score, ":\n"
	    print clf.best_score_
	    Y_true, Y_pred  = test["class"], clf.predict(gwbowv_test)
	    print "Report"
	    print classification_report(Y_true, Y_pred, digits=6)
	    print "Accuracy: ",clf.score(gwbowv_test,test["class"])
	    print "Time taken:", time.time() - strt, "\n"
	endtime = time.time()
	print "Total time taken in training: ", endtime-start, "seconds." 

	print "********************************************************"
