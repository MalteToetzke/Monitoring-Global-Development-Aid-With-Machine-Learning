import pandas as pd
import os

####
"""Map the cluster back to aid activities via labels (generated in "Trainings_df.py")"""
####

def csv_import(name, dtype=str, delimiter="|"):
    x = pd.read_csv(name, encoding='latin-1', dtype=dtype, low_memory=False, delimiter=delimiter, index_col=False)
    return x

#load data
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#load data
#clusters and labels
clusters=df = csv_import(os.path.join(dir_path, 'Data/clusters_KM_178_clean.csv'),dtype={'cluster': float})
#projects and labels
df=csv_import(os.path.join(dir_path, 'Data/labeled_projects.csv'), dtype={'PurposeCode': float})

df['cluster'] = ''
df['ClusterTitle']=''
print('start loop')

for i, row in df.iterrows():
    label = row['label']
    df.at[i, 'cluster'] = clusters[clusters['label'] == label].reset_index(drop=True)['cluster'][0]
    df.at[i, 'ClusterTitle'] = clusters[clusters['label'] == label].reset_index(drop=True)['title'][0]

# save dataframe with development aid activity data and activity clusters
df.to_csv("projects_clusters_complete.csv", index=False, header=True, sep='|')


