# WordVectors
Text Classification with Word Vectors


# Download Pre-trained Vectors
* cd into Repository
* wget http://nlp.stanford.edu/data/glove.6B.zip
* unzip glove.6B.zip

# Install
* pip install pandas sklearn gensim tensorflow keras bs4
* nltk.download('punkt')
* nltk.download('stopwords')

# Get Reduced Vectors from Pre-trained vectors
* python reduction_algo.py [embedding_file] [reduced_dimensions] (e.g. python reduction_algo glove.300d.txt 150)
* e.g. python reduction_algo glove.300d.txt 150 --> the reduced embeddings will be saved in reduced_embeddings_150.txt

# 20Newsgroup on Pretrained-Glove
* svc.py creates the document vectors + gives results
* e.g. python svc.py glove.300d.txt 300

# Reuters on Pretrained Glove
* svc_reuters.py creates the document vectors, metrics.py gives the results
* e.g. python svc_reuters.py glove.300d.txt 300 then, python metrics.py 300

# Train Vectors Using Word2Vec Model
* Run the Word2VecModel_on_Newsgroup.py and Word2VecModel_on_Reuters.py files
* Embedding files will be created, use them just as pre-trained vectors for evaluation

# Evaluation Table

| Embedding | 20Newsgroup | Reuters |
| :---         |     :---:      |          ---: |
| Glove-300D   |   60   |     |
| Glove-200D     |   53     |       |
| Glove-100D   |  50   |     |
| Glove-Reduced-150D     |   51     |       |
| Glove-Reduced-100D     |   42     |       |
| Glove-Reduced-50D     |    36    |       |
| Fasttext-300D          |      |     |
| Fasttext-Reduced-150D     |        |       |
| Word2Vec-300D          |      |     |
| Word2Vec-Reduced-150D     |        |       |
| W2V-Newsgroup-300D     |   73 (0.7379182156133829)     |   x    |
| W2V-Newsgroup-200D     |   0.6736590546999469     |   x    |
| W2V-Newsgroup-400D     |  0.7124269782262347      |   x    |
| W2V-Newsgroup-Reduced-150D     |  60 (0.6023632501327668)      |   x    |
| W2V-Newsgroup-Reduced-100D     |        |   x    |
| W2V-Newsgroup-Reduced-200D     |  0.6427243759957515      |   x   |
| W2V-Reuters-300D     |   x    |  41 (0.4121083377588954)      |
| W2V-Reuters-200D     |   x     |       |
| W2V-Reuters-100D     |   x     |       |
| W2V-Reuters-Reduced-150D     |   x      |  32 (0.3252788104089219)     |
| W2V-Reuters-Reduced-100D     |   x     |       |
| W2V-Reuters-Reduced-50D     |    x    |       |
