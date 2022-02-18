import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import re
import os


#####
"""Preprocess textual descriptions for document embedding. Reduce vocabulary through lemmatization, stopword removal, lowercasing """
#####


"""necessary helper functions:"""
#Noise cancelling in text
def scrub_words(text):
    """Basic cleaning of texts."""

    # remove html markup
    text = re.sub("(<.*?>)", "", text)

    # remove other weird letters
    text = re.sub(r"\w*[□,©,$]\w*", "", text)

    # remove non-ascii and digits
    text = re.sub("(\\W|\\d)", " ", text)

    # remove whitespace
    text = text.strip()
    return text

# remove whitespaces
def remove_whitespace(x):
    """
    Helper function to remove any blank space from a string
    x: a string
    """
    try:
        # Remove spaces inside of the string
        x = " ".join(x.split())

    except:
        pass
    return x

#import dataframe
def csv_import(name, dtype=str, delimiter="|"):
    x = pd.read_csv(name, encoding='latin-1', dtype=dtype, low_memory=False, delimiter=delimiter)
    return x


stopwords=["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
print("starting")

if __name__ == '__main__':
    # Define a parameter structure args
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(dir_path)
    args = {  # model parameters:
        #'model_name': 'baseline'  # any parameters
    }
    import_data=True
    preprocess_data=True

    if import_data:
        df = csv_import(os.path.join(dir_path, 'Data/trans_subset_training2.csv'),
                        dtype={'raw_text': str, "PurposeCode": float})
        df.raw_text = df.raw_text.astype(str)
        df = pd.DataFrame(df)

    if preprocess_data:
        # Lowercasing
        df['raw_text'] = [word.lower() for word in df['raw_text']]

        # Stop Word Removal
        df['raw_text'] = df['raw_text'].str.lower().str.split()
        df['raw_text'] = df['raw_text'].apply(lambda x: [item for item in x if item not in stopwords])

        # Lemmatization & Noise removal in one step
        #For lsf cluster: download manually in command prompt with: python -m nltk.downloader 'wordnet'
        nltk.download('wordnet')

        # init lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Function
        for i, row in df.iterrows():
            if i%20000==0:
                print(i)
            # Lemmatize
            lem_text = [lemmatizer.lemmatize(word=word, pos='v') for word in row['raw_text']]
            # remove Noise
            clean_text = [scrub_words(w) for w in lem_text]
            df.at[i, 'raw_text'] = clean_text

        # Join back for Text
        df['raw_text'] = df['raw_text'].str.join(' ')
        print(df.shape)
        print(df.loc[20,'raw_text'])
        df['raw_text'] = df['raw_text'].apply(remove_whitespace)

        df.to_csv("preprocessed_subset_training.csv", index=False, header=True, sep='|')


