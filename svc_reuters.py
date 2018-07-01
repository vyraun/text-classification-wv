import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility
from numpy import float32
import math
from sklearn.ensemble import RandomForestClassifier
import sys
from random import uniform
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
import pickle
import cPickle
from math import *
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn import grid_search
from sklearn.mixture import GMM
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
import pandas as pd
import time
from nltk.corpus import stopwords
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
		    document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove[word]
		else:
		    document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove['unk']
		    	
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
		    document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove[word]
		else:
		    document_vector += (word_to_tf[word]*word_idf_dict[word])*Glove['unk']
		    	
	# These words will be passed to generate the document vector
	gwbowv[counter] = document_vector # Code to construct the document vector
	
	counter+=1
	if counter % 1000 == 0:
		print "Test Text Covered : ",counter
		#print "Document Vector", document_vector

#saving gwbowv train and test matrices
#np.save('Reuters_train_{}'.format(dims), gwbowv)
#np.save('Reuters_test_{}'.format(dims), gwbowv_test)

endtime = time.time() - start
print "Total time taken: ", endtime, "seconds." 

print "********************************************************"

print "Fitting One vs Rest SVM classifier to labeled cluster training data..."

start = time.time()

print("Loading Training and Test Data from Pickled file")

#all = pd.read_pickle('all.pkl')

# Get train and text data.
lb = MultiLabelBinarizer()
Y = lb.fit_transform(all.tags)
train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

train = DataFrame({'text': []})
test = DataFrame({'text': []})

train["text"] = train_data.reset_index(drop=True)
test["text"] = test_data.reset_index(drop=True)

#print("Sample Training data")
#print(train.head(2))

#print("Training Y")
#print(Y_train[9])

print("Loading Document Vectors")

# Load feature vectors.
#gwbowv = np.load('Reuters_train_{0}.npy'.format(dims))

#print("Sample Doc Vectors")
#print(gwbowv[4])

#gwbowv_test = np.load('Reuters_test_{0}.npy'.format(dims))

#clf = OneVsRestClassifier(LogisticRegression(C = 100.0))
#clf = clf.fit(gwbowv, Y_train)
#pred = clf.predict(gwbowv_test)

print("Model Training Starts")

param_grid = [
  {'estimator__C': np.arange(0.1, 150, 10)}
  # {'C': [1, 10, 100, 200, 500, 1000], 'gamma': [0.01, 0.05, 0.001, 0.005,  0.0001], 'kernel': ['rbf']},
 ]
scores = ['f1_weighted'] #, 'accuracy', 'recall', 'f1']
for score in scores:
    strt = time.time()
    print "# Tuning hyper-parameters for", score, "\n"
    clf = GridSearchCV(OneVsRestClassifier(LogisticRegression(C = 100.0), n_jobs=1), param_grid, cv=5, scoring= '%s' % score) # n_jobs=30
    clf = clf.fit(gwbowv, Y_train)
    
    pred = clf.predict(gwbowv_test)
    pred_proba = clf.predict_proba(gwbowv_test)

    K = [1,3,5]

    for k in K:
        Total_Precision = 0
        Total_DCG = 0
        norm = 0
        for i in range(k):
            norm += 1/math.log(i+2)

        loop_var = 0
        for item in pred_proba:
            classelements = sorted(range(len(item)), key=lambda i: item[i])[-k:]
            classelements.reverse()
            precision = 0
            dcg = 0
            loop_var2 = 0
            for element in classelements:
                if Y_test[loop_var][element] == 1:
                    precision += 1
                    dcg += 1/math.log(loop_var2+2)
                loop_var2 += 1
            
            Total_Precision += precision*1.0/k
            Total_DCG += dcg*1.0/norm
            loop_var += 1
        print "Precision@", k, ": ", Total_Precision*1.0/loop_var 
        print "NDCG@", k, ": ", Total_DCG*1.0/loop_var

    print "Coverage Error: ", coverage_error(Y_test, pred_proba)
    print "Label Ranking Average precision score: ", label_ranking_average_precision_score(Y_test, pred_proba) 
    print "Label Ranking Loss: ", label_ranking_loss(Y_test, pred_proba)
    print "Hamming Loss: ", hamming_loss(Y_test, pred)
    print "Weighted F1score: ", f1_score(Y_test, pred, average = 'weighted')

    # print "Total time taken: ", time.time()-start, "seconds."


endtime = time.time()
print "Total time taken: ", endtime-start, "seconds." 

print "********************************************************"
