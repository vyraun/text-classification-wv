# WordVectors
Text Classification with Word Vectors

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

pip install pandas sklearn gensim tensorflow keras bs4
nltk.download('punkt')
nltk.download('stopwords')

* 20ewsGroup: svc creates the document vectors + gives results
* Reuters: svc_reuters creates the document vectors, metrics gives the results


* reduction_algo.py embed_file -> second arg must be the wanted dimensions
* and the file name of the saved embedding must mention the new dimensions
