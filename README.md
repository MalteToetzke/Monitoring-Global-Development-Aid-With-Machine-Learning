# Monitoring-Global-Development-Aid-With-Machine-Learning
This repository contains the underlying code for the paper "Monitoring Global Development Aid with Machine Learning" by Toetzke, M.; N. Banholzer; and S. Feuerriegel.


## Code: 

### Unsupervised categorization of Development Aid Activities based on their textual descriptions using document embedding and clustering
Includes preprocessing of texts, training of Paragraph Vector model and clustering of resulting document vectors.

### Requirements #
* pandas ==0.25.1
* numpy ==1.15.4
* spacy >=2.2.1
* nltk >=3.4.5
* scikit-learn >=0.21.3
* gensim >=3.8.0

### Usage #
Run scripts in the following order:

Preprocessing
* word_preprocess.py (Preprocessing of texts using stopword removal, lowercasing, lemmatization)
* Clear_regions (replace words referring to region our country - by named entity recognition (spacy))
* Trainings_df.py (create label for each unique textual description, in new df drop all texts after occuring the second time)

Document embedding
* Embedding.py (2-phase training of Parameter Vector model: Wikipedia corpus and aid activity descriptions descriptions)
* Create_vectors.py (Derive document vectors from model)
* Closest_vectors.py (test model result by requesting closest text for random document)

Clustering
* Normalize.py (Make vectors unit vectors to so cosine distance=eucl. distance)
* Clustering_crossvalidation.py (Cross_validate hyperparams based on silhoutte and elbow diagram to find optimal number of clusters n)
* Clustering_optimal_n.py (Compute clusters based on optimal n)
* Cluster_topic (Compute tf-idf weights for all words in cluster, derive 20 words with highest tf-idf for each cluster)
* Backmapping_clusters.py (Map clusters to aid activities)

## Data: 
### Data to perform analyses on development aid activities across topics, time, and recipient countries. 
