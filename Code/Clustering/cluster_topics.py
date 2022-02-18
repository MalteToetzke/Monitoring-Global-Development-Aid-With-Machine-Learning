import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

####
"""Get most characteristic words of clusters via tfidf: For analyzing cluster topics"""
####

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

def csv_import(name, dtype=str, delimiter="|"):
    x = pd.read_csv(name, encoding='latin-1', dtype=dtype, low_memory=False, delimiter=delimiter, index_col=False)
    return x

#load data
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df = csv_import(os.path.join(dir_path, 'Data/clusters_KM_178.csv'),dtype={'cluster': float})
print(df.shape)  ##should be 380936,4

df=df[['cluster','text']]
print(df.shape)

#join all texts of same cluster to one large text per cluster
df=df.groupby(['cluster'])['text'].apply(lambda x: ' '.join(x)).reset_index()

print('shape of cluster df is:')
print(df.shape)

texts=df['text']

tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=20, max_df=0.8)

# fit combined vector
tf_idf_object = tfidf_vectorizer.fit_transform(texts)
#link words to features
feature_names = tfidf_vectorizer.get_feature_names()

df['topic']=' '

print(tf_idf_object.shape[0])
print(df.shape)

for i in range(tf_idf_object.shape[0]):
    vector = tf_idf_object[i]

    # sort vector by tf_idf
    sorted_vector = sort_coo(vector.tocoo())

    # extract only the top n; n here is 20
    keywords = extract_topn_from_vector(feature_names, sorted_vector, 20)
    keywords = pd.DataFrame(list(keywords.items()), columns=['Words', 'Values'])

    keywords = keywords.sort_values(by=["Values"], ascending=False)

    # Get only the ordered words
    keywords = keywords['Words']
    # make list
    keywords = keywords.tolist()
    #number of words in sentence
    number = len(keywords)

    # join list of words to sentence /single string
    text = ' '.join(word for word in keywords)

    # Add string to df
    df.at[i, 'topic'] = text


print('test how topics look like')
print(df.loc[0,'topic'])
print(df.loc[10,'topic'])

print('shape of cluster_topics df is:')
print(df.shape)
df=df[['cluster','topic']]
df.to_csv("cluster_topics_KM_178_clean.csv", index=False, header=True)