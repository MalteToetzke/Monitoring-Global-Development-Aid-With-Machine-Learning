
import pandas as pd
import os

#####
""" Drop duplicate textual descriptions from training set. Assign a label to each textual description that maps to aid activity """
#####


def csv_import(name, dtype=str, delimiter="|"):
    x = pd.read_csv(name, encoding='latin-1', dtype=dtype, low_memory=False, delimiter=delimiter)
    return x


#Load Data

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

df = csv_import(os.path.join(dir_path, 'Data/preprocessed_df.csv'),
                        dtype={"PurposeCode": float})
df.raw_text = df.raw_text.astype(str)
df['forTraining']=''

for i,row in df.iterrows():

    # make list
    keywords = row['raw_text'].split()
    #number of words in sentence
    number = len(keywords)

    # Decide whether to use for embedding training or not --> number of words must be >5
    if number > 5:
        df.at[i,'forTraining'] = 'yes'
    else:
        df.at[i, 'forTraining'] = 'no'


print('shape of projects is:')
print(df.shape)

print('data used for training is')
for_training=df[df['forTraining']=='yes']
print(for_training.shape)

## assign a unique category to each TEXT. category maps back to aid activity that contains the text.
df=df.assign(label=(df['raw_text'].astype(str)).astype('category').cat.codes)
df['label']='project' + df['label'].astype(str)
print(df.shape)
print('save dataframe with labels')
df.to_csv("labeled_projects.csv", index=False, header=True, sep='|')
##drop doubles in project descriptions
df=df.drop_duplicates(subset='label', keep='first')
print(df.shape)
df=df[df['forTraining']=='yes']
print(df.shape)
print('save data for training')
df.to_csv("subset_training.csv", index=False, header=True, sep='|')

