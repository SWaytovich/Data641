import nltk
import numpy as np
import pandas as pd
from collections import Counter
import liwc
import string
#Initial build:
#Ignore URLs

#Later updates:
#Deal with URL, named entities? Perhaps count these as separate category?
#Look through dictionary for other bigrams to merge

#merging words for specific references in LIWC:
tk = nltk.tokenize.WhitespaceTokenizer()
tokenizer = tk.tokenize

def merge_words(text):
    temp = text.replace("kind of","kindof")
    return temp

def remove_punctuation(text):
    punclist = string.punctuation
    punclist = punclist.replace('\'','')
    text = ''.join(c for c in text if c not in punclist)
    #text = text.translate(str.maketrans('','',string.punctuation))
    return text

def clean_text_liwc(text):
    lower = text.lower()
    merged = merge_words(lower)
    output = remove_punctuation(merged)
    return output

def load_categories(filename):
    catdf = pd.read_csv(filename,header=None)
    return list(catdf.iloc[:,0])

def liwc_debugger(text,tokenizer):
    cleaned = clean_text_liwc(text)
    tokenized = tokenizer(cleaned)
    tokenized = [remove_punctuation(token) for token in tokenized]
    return tokenized


def liwc_vectorizer(text,tokenizer,categories,parse):
    cleaned = clean_text_liwc(text)
    tokenized = tokenizer(cleaned)
    tokenized = [remove_punctuation(token) for token in tokenized]
    wordct = len(tokenized)
    counts = Counter(category for token in tokenized for category in parse(token))
    vecsize = len(categories)
    outvec = np.zeros(vecsize,dtype=int)
    for i in range(0,vecsize):
        outvec[i] = counts[categories[i]]
    return outvec,wordct

#Converting text to pandas series to dataframe with output vectors
def liwc_series_vectorizer(inputseries,tokenizer,categories,parse):
    templist = [liwc_vectorizer(line,tokenizer,categories,parse) for line in inputseries]
    normalized = [line[0]/line[1] for line in templist] #dividing by total word count
    tdf = pd.DataFrame(normalized,columns =categories)
    wordcts = [line[1] for line in templist]
    tdf["WordCount"] = pd.Series(wordcts)
    return tdf
