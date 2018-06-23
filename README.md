# WordVectors
Text Classification with Word Vectors


# Download Pre-trained Vectors
* wget http://nlp.stanford.edu/data/glove.6B.zip
* unzip glove.6B.zip

# Install
* pip install pandas sklearn gensim tensorflow keras bs4
* nltk.download('punkt')
* nltk.download('stopwords')

# Get Reduced Vectors from Pre-trained vectors
* reduction_algo.py embedding_file reduced_dimensions (e.g. python reduction_algo glove.300d.txt 150)
* e.g. python reduction_algo glove.300d.txt 150 --> the reduced embeddings will be saved in reduced_embeddings_150.txt

# 20Newsgroup
* svc.py creates the document vectors + gives results
* e.g. python svc.py glove.300d.txt 300


# Reuters 
* svc_reuters.py creates the document vectors, metrics.py gives the results
* e.g. python svc_reuters.py glove.300d.txt 300 then, python metrics.py 
