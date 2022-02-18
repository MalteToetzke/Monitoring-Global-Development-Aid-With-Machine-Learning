import pandas as pd
import os
from gensim.models.doc2vec import Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np

####
"""Extract the document vectors from the doc2vec model. Attribute them to labels that can be mapped to aid activities"""
###

class EpochLogger(CallbackAny2Vec):
#'''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))


def csv_import(name, dtype=str, delimiter="|"):
    x = pd.read_csv(name, encoding='latin-1', dtype=dtype, low_memory=False, delimiter=delimiter)
    return x
#load data
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

df = csv_import(os.path.join(dir_path, 'Data/subset_long.csv'))

##build lists from training:
##only use those projects that have forTraining=="yes"
text=df['raw_text']
label=df['label']

text=text.reset_index(drop=True)
label=label.reset_index(drop=True)
label=label.rename(columns={'0': 'label'})
print('lets test how these texts look like')
print(text[1])
print(label.shape)

vectors=[]


print('build dataframe for clustering')
label=pd.DataFrame(label)
label['label']=label
print(label['label'].head)
label['text'] = text
print(label.shape)

tags_projects=label['label']
#Load the model
model = Doc2Vec.load(os.path.join(dir_path, 'Data/model_DevFund.doc2vec'))

for i in range(len(tags_projects)):
    vectors.append(model.docvecs[tags_projects[i]])
    if i%100000==0:
        print(i)
        print(tags_projects[i])


vectors=pd.DataFrame(vectors)
print('saving to disc')
#deleted_labels.to_csv("deleted_labels.csv", header=False,index=False, sep='|')
vectors.to_csv("vectors.csv", header=False,index=False, sep='|')
label.to_csv("label_text.csv", index=False, header=True, sep='|')