import numpy as np
import sys
import cPickle as pickle
from sklearn.decomposition import PCA
import subprocess
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

filename = sys.argv[1]
dims = int(sys.argv[2])
red_dims = int(sys.argv[3])

model = Word2Vec.load(filename)
words = list(model.wv.vocab)

Glove = {}

print("Loading Vectors.")
for word in words:
    coefs =  model.wv[word]
    Glove[word] = coefs

print("Done.")

X_train = []
X_train_names = []
for x in Glove:
        X_train.append(Glove[x])
        X_train_names.append(x)

X_train = np.asarray(X_train)
pca_embeddings = {}

# PCA to get Top Components
pca =  PCA(n_components = dims)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
U1 = pca.components_

z = []

# Removing Projections on Top Components
for i, x in enumerate(X_train):
	for u in U1[0:7]:        
        	x = x - np.dot(u.transpose(),x) * u 
	z.append(x)

z = np.asarray(z)

# PCA Dim Reduction
pca =  PCA(n_components = red_dims)
X_train = z - np.mean(z)
X_new_final = pca.fit_transform(X_train)


# PCA to do Post-Processing Again
pca =  PCA(n_components = red_dims)
X_new = X_new_final - np.mean(X_new_final)
X_new = pca.fit_transform(X_new)
Ufit = pca.components_

X_new_final = X_new_final - np.mean(X_new_final)

final_pca_embeddings = {}
filename_reduced = "reduced_embeddings_on_trained_{}.txt".format(red_dims)
embedding_file = open(filename_reduced, 'w')

for i, x in enumerate(X_train_names):
        final_pca_embeddings[x] = X_new_final[i]
        embedding_file.write("%s\t" % x)
    	for u in Ufit[0:7]:
            final_pca_embeddings[x] = final_pca_embeddings[x] - np.dot(u.transpose(),final_pca_embeddings[x]) * u 

        for t in final_pca_embeddings[x]:
                embedding_file.write("%f\t" % t)
        
        embedding_file.write("\n")
	
# Placing a 'unk' token with zero values
embedding_file.write("unk\t")
unk = np.zeros(300, dtype='float32')
for t in unk:
    embedding_file.write("%f\t" % t)
embedding_file.write("\n")

print("The Reduced Embedding is available at {0}".format(filename_reduced))

#print("Results for the Embedding")
#print subprocess.check_output(["python", "all_wordsim.py", "pca_embed2.txt", "data/word-sim/"])
#print("Results for Glove")
#print subprocess.check_output(["python", "all_wordsim.py", "../glove.6B/glove.6B.300d.txt", "data/word-sim/"])
