from sklearn import preprocessing  # to normalise existing X
import os
import pandas as pd

###
"""To normalize vectors to unit size --> to get cosine distance for clustering"""
###

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
X = pd.read_csv(os.path.join(dir_path, 'Data/vectors.csv'), dtype=float, delimiter='|',header=None, index_col=False )
print(X.shape)
X_Norm = preprocessing.normalize(X)
X_Norm=pd.DataFrame(X_Norm)
print(X_Norm.shape)
print(print(X_Norm.loc[0,:]))
X_Norm.to_csv("unit_vectors.csv", header=False,index=False, sep='|')
