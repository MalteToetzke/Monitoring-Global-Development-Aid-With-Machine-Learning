import re
import pandas as pd
import os
import spacy
nlp = spacy.load("en_core_web_sm")

####
"""Remove mentions of countries, cities, or populations from textual descriptions"""
####


def csv_import(name, dtype=str, delimiter="|"):
    x = pd.read_csv(name, encoding='latin-1', dtype=dtype, low_memory=False, delimiter=delimiter)
    return x


#load data
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

df = csv_import(os.path.join(dir_path, 'Data/preprocessed_df.csv'))
df.raw_text = df.raw_text.astype(str)
regions=['GPE', 'NORP', 'LOC']


def replace_person_names(token):
    if token.ent_iob != 0 and token.ent_type_ in regions:
        return token.ent_type_.lower()+ ' '
    return token.string

def redact_names(nlp_doc):
    for ent in nlp_doc.ents:
        ent.merge()
    tokens = map(replace_person_names, nlp_doc)
    return ''.join(tokens)

shortword = re.compile(r'\W*\b\w{1,1}\b')

for i, row in df.iterrows():
    df.at[i,'raw_text']=redact_names(nlp(row['raw_text']))
    df.at[i, 'raw_text']=shortword.sub('', row['raw_text'])

print(df.shape)

df.to_csv("preprocessed_df.csv", index=False, header=True, sep='|')
