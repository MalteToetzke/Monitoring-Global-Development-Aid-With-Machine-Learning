##https://markroxor.github.io/gensim/static/notebooks/doc2vec-wikipedia.html
##https://dumps.wikimedia.org/enwiki/
##https://radimrehurek.com/gensim/corpora/wikicorpus.html
import gensim

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec
from nltk.stem import WordNetLemmatizer
import nltk
import os
import pandas as pd
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
text=text.str.split()
label=label.reset_index(drop=True)

text.to_csv("texts_all.csv", index=False, header=True, sep='|')
print('texts saved to disk')
label.to_csv("labels_all.csv", index=False, header=True, sep='|')
print('labels saved to disk')

print('start wikipedia')

"""Dowload wikipedia corpus for pretraining"""
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

wiki = WikiCorpus(os.path.join(dir_path, 'Data/enwiki-latest-pages-articles.xml.bz2'))

##Irgnore short articles
wiki.ARTICLE_MIN_WORDS = 50  ##can set higher (e.g. to 50)


## Define: Convert WikiCorpus to suitable Doc2Vec form --> test if this fits for preprocessing too!
class PreprocessWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            words.append([lemmatizer.lemmatize(word=c, pos='v') for c in content])
            tagged.append(title)


class EpochLogger(CallbackAny2Vec):
#'''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

##Use same lemmatization as for projects to ensure that vocab is the same!
words=[]
tagged=[]
PreprocessWikiDocument(wiki)
for content, (page_id) in wiki.get_texts():
    words.append([lemmatizer.lemmatize(word=c, pos='v') for c in content])
    tagged.append(page_id)
    if len(words)%1000000==0:
        print(len(words))

class TaggedDocumentIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc, tags=[self.labels_list[idx]])

print('test how wiki texts look like:')
print('element 1:')
print(words[1])

print('create list out of the dataframe texts and labels') #new
text=list(text)
#print(text[0])
label=list(label)
#print(label[0])


print('merging the two lists together for vocabulary') #new
words2=words+text
tagged2=tagged+label
print('length of wiki corpus is:')
print(len(tagged))
##all documents(to initiate vocabulary AND labels for all docs

#to build total vocab and labels
print('document iterator starts')
# both wikipedia and aid activity descriptions
documents2 = TaggedDocumentIterator(words2, tagged2)



#print('length of total corpus is:')
#print(len(documents2))
print('and the elements of wiki look like:')
print(documents2.doc_list[1])
##only wikipedia for first training
documents=TaggedDocumentIterator(words, tagged)

##for project training
# only textual aid descriptions
documents3=TaggedDocumentIterator(text, label)

print('and the elements of projects look like:')
print(documents3.doc_list[1])
##vector size 300 should be fine. window size 6



#define Model ---check for alternatives
# PV-DBOW
cores = multiprocessing.cpu_count()
model=Doc2Vec(dm=0, dbow_words=1, size=200, window=6, min_count=30,iter=10,workers=cores)

model.build_vocab(documents2)
print('vocab built')



epoch_logger = EpochLogger()
# First training only on wikipedia data
model.train(documents,total_examples=model.corpus_count, epochs=model.iter, callbacks=[epoch_logger])
# Second training on activity descriptions
model.train(documents3,total_examples=model.corpus_count, epochs=25, callbacks=[epoch_logger])
# Store the model to mmap-able files
model.save('model_DevFund.doc2vec')