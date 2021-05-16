import pandas as pd
import numpy as np
from final_proj_funcs import *
import tensorflow as tf
from tensorflow.keras import (
    models, losses, optimizers,
    layers, preprocessing,
)
from sklearn.model_selection import (
    train_test_split, KFold,
    StratifiedKFold, cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn import tree, ensemble, svm
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score,confusion_matrix
)
import unicodedata
import spacy
import os

import nltk,re
from liwc_funcs import *
import liwc
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def whitespace_tokenizer(line):
    return line.split()

def remove_accents(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', \
        'ignore').decode('utf-8', 'ignore')
    return new_text

def remove_encoding(text):
    new_text = text.replace('\x92', "'")
    new_text = new_text.replace('\x94', '"')
    new_text = new_text.replace('\x93', '"')
    return new_text


stop_file = 'mallet_en_stoplist.txt'
proc_file = 'new_legis_proc_jargon_stopwords.txt'

stop_words = load_stopwords(stop_file)
proc_words = load_stopwords(proc_file)

df = pd.read_csv('wcpr_mypersonality.csv',encoding="ISO-8859-1")

#df['Target'] = df['cNEU']
df['Target'] = pd.Series(np.where(df['cNEU'] == 'y', 1, 0))

target = df['Target']
status = df[['STATUS' ,'#AUTHID']]
dist_users = df[['#AUTHID', 'Target']].drop_duplicates()

newdf = df.groupby('#AUTHID').agg({'STATUS': lambda x: ' '.join(x), 'NETWORKSIZE': 'mean', 'BETWEENNESS':'mean',
       'NBETWEENNESS':'mean', 'DENSITY':'mean', 'BROKERAGE':'mean', 'NBROKERAGE':'mean', 'TRANSITIVITY':'mean',
       'Target':'mean'})

#cleaning status data
newdf['STATUS'] = newdf['STATUS'].apply(remove_accents).apply(remove_encoding)

#ADDING OTHER VARIABLES TO REGRESSION
newx = newdf[['STATUS','NETWORKSIZE', 'BETWEENNESS', 'NBETWEENNESS', 'DENSITY',
        'BROKERAGE', 'NBROKERAGE', 'TRANSITIVITY']]

#Performing train-test split at the user level
x_train, x_test, y_train, y_test = train_test_split(newx, newdf['Target'],\
    test_size=0.3, random_state=4, stratify=dist_users['Target'])

# #Reconstructing training and test sets at the status post level
# train_data = pd.concat([x_train, y_train], axis=1)
# test_data = pd.concat([x_test, y_test], axis=1).sort_values(by=['#AUTHID'])
# x_train_feat = pd.merge(status, train_data, on='#AUTHID',how='inner')
# x_test_feat = pd.merge(status, test_data, on='#AUTHID',how='inner')
#
# # y_test_label = pd.concat([x_test_feat['#AUTHID'], y_test])


###############################################################################
#BIGRAM COUNT VECTORIZER:
#takes a pandas series and vectorizes it based on bigram count,
#also outputs vectorizer to replicate transformation on other series
def bigram_series_vectorizer(x_train_series,stop_words,proc_words,remove_stopword_bigrams=True):
    #Converting to unigram&bigram token counts
    x_train_strings = convert_lines_to_feature_strings(x_train_series, stop_words, \
        proc_words, remove_stopword_bigrams)
    x_train_vec, training_vectorizer = convert_text_into_features(x_train_strings, \
        stop_words, whitespace_tokenizer)
    #Constructing vectorizer function to replicate transformation
    def bigram_vectorizer(x_series):
        x_series_strings = convert_lines_to_feature_strings(x_series, stop_words, \
            proc_words, remove_stopword_bigrams)
        x_series_vec = training_vectorizer.transform(x_series_strings)
        return x_series_vec
    return x_train_vec,bigram_vectorizer

x_train_bigm,vectorizer_bigm = bigram_series_vectorizer(x_train["STATUS"],stop_words,proc_words,remove_stopword_bigrams=True)
x_test_bigm = vectorizer_bigm(x_test["STATUS"])

##############################################################################
# LIWC VECTORIZER:
LIWC_file = 'myliwc.dic'
tk = nltk.tokenize.WhitespaceTokenizer()
tokenizer = tk.tokenize

#Loading LIWC dictionary
parse,categories = liwc.load_token_parser(LIWC_file)
#vector will include all categories in LIWC dictionary
#can also import custom list of specific categories to count:
#categories = load_categories("categorylist.txt")

x_train_LIWC = liwc_series_vectorizer(x_train["STATUS"],tokenizer,categories,parse)
x_test_LIWC = liwc_series_vectorizer(x_test["STATUS"],tokenizer,categories,parse)
##############################################################################
#### MODEL TRAINING AND OUTPUT (BIGRAM)

mod_bigm = LogisticRegression(solver='liblinear')
mod_bigm.fit(x_train_bigm, y_train)
y_hat_bigm = pd.Series(mod_bigm.predict(x_test_bigm))

print("Accuracy/Precision/Recall for Bigram Count Vectorization:")
print(accuracy_score(y_test, y_hat_bigm))
print(precision_score(y_test, y_hat_bigm))
print(recall_score(y_test, y_hat_bigm))


# #############################################################################
# # MODEL TRAINING AND OUTPUT (LIWC)
mod_LIWC = LogisticRegression(solver='liblinear')
mod_LIWC.fit(x_train_LIWC, y_train)
y_hat_LIWC = pd.Series(mod_LIWC.predict(x_test_LIWC))

print("Accuracy/Precision/Recall for LIWC:")
print(accuracy_score(y_test, y_hat_LIWC))
print(precision_score(y_test, y_hat_LIWC))
print(recall_score(y_test, y_hat_LIWC))
