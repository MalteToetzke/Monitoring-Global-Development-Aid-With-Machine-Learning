from sklearn.cluster import KMeans
import pandas as pd
import os
import pickle

####
"""Based on cross-validated optimal number of clusters perform clustering task and assign clusters to descriptions"""
####

def csv_import(name, dtype=str, delimiter="|"):
    x = pd.read_csv(name, encoding='latin-1', dtype=dtype, low_memory=False, delimiter=delimiter, index_col=False)
    return x

#load data
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df = csv_import(os.path.join(dir_path, 'Data/label_text2.csv'))
vec = pd.read_csv(os.path.join(dir_path, 'Data/unit_vectors2.csv'), dtype=float, delimiter='|',header=None, index_col=False)

print('initilizing k-means++ with 60 initializations')
km = KMeans(
    init='k-means++',
    n_clusters=178, n_init=100, max_iter=1000,   ###change the number of clusters!
        tol=1e-07, random_state=80
    )

print('start fitting and assigning clusters')
df['cluster']=km.fit_predict(vec)

print('save the model')
pkl_filename = "k_means.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(km, file)

distortion=km.inertia_
print('the distortion is:')
print(distortion)


##save tthe df with clusters
print('saving df with clusters')
df.to_csv("clusters_KM.csv", index=False, header=True, sep='|')
