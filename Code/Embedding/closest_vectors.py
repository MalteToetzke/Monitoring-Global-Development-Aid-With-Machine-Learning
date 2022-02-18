from scipy.spatial import distance
import pandas as pd
import os
from scipy import spatial
import numpy as np

####
"""This is to check the plausibility of the embedding: Get closest documents for any embedded document"""
####


def csv_import(name, dtype=str, delimiter="|"):
    x = pd.read_csv(name, encoding='latin-1', dtype=dtype, low_memory=False, delimiter=delimiter, index_col=False)
    return x

#load data
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#load data
df_v=pd.read_csv(os.path.join(dir_path, 'Data/vectors.csv'), dtype=float, delimiter='|',header=None, index_col=False )
df = csv_import(os.path.join(dir_path, 'Data/label_text.csv'))
vectors=df_v.to_numpy()

#target text
random_text_index=45400  ## any number in range
target=vectors[random_text_index,:]
print(df.iloc[random_text_index]['text'])

pd.options.display.max_colwidth = 500


# Find the distance between this node and everyone else
cosine_distances = df_v.apply(lambda row: spatial.distance.cosine(row, target), axis=1)

# Create a new dataframe with distances.
distance_frame = pd.DataFrame(data={"dist": cosine_distances, "idx": cosine_distances.index})

distance_frame.sort_values("dist", inplace=True)
print(distance_frame.shape)
print(distance_frame.head)
# nodes
smallest_dist_ixs = distance_frame.iloc[1:15]["idx"]
print(smallest_dist_ixs)
print(distance_frame.iloc[1:20]['dist'])
most_similar_nodes = df.loc[smallest_dist_ixs,'text']
print(most_similar_nodes)
