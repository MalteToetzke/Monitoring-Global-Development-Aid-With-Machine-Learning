import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys

####
"""Plot silhouette score and distortion to cross-validate optimal cluster number"""
####

#load data
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

vec = pd.read_csv(os.path.join(dir_path, 'Data/unit_vectors.csv'), dtype=float, delimiter='|',header=None, index_col=False)


##Clustering of selected points
distortions = []
km_silhouette = []

#range of cluster numbers to validate
clusters=range(150,250)
for i in clusters:
    print(i)#later use range from (250, 280)
    km = KMeans(
        init='k-means++', n_clusters=i,
        n_init=10, max_iter=300,
        tol=1e-04, random_state=80
    )
    km.fit(vec)
    preds = km.predict(vec)
    ##for kmeans ellbow diagram
    distortions.append(km.inertia_)

    #for silhouette diagramme
    silhouette = silhouette_score(vec, preds)
    km_silhouette.append(silhouette)

print(distortions)
print(km_silhouette)


#make plots
def plotting(y_data,x_values,name_y,name_plot, number):
    f=plt.figure(number)
    plt.plot(x_values, y_data, marker='o')  ##later use range from (250, 280)
    plt.xlabel('Number of clusters')
    plt.ylabel(name_y)
    f.savefig(name_plot, format='png', bbox_inches='tight')
    #plt.show()

# plot kmeans distortion
plotting(distortions,clusters, name_y='Distortion',name_plot='ellbow.png', number=1)

# plotting silhouette
plotting(km_silhouette,clusters, name_y='Silhoutte_score',name_plot='silhouette.png', number=2)


