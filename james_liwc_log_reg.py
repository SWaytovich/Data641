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
    confusion_matrix
)
import spacy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def whitespace_tokenizer(line):
    return line.split()

import nltk,re
from liwc_funcs import *
import liwc

stop_file = 'mallet_en_stoplist.txt'
#proc_file = '../new_legis_proc_jargon_stopwords.txt'
LIWC_file = 'myliwc.dic'

stop_words = load_stopwords(stop_file)
#proc_words = load_stopwords(proc_file)

df = pd.read_csv('wcpr_mypersonality.csv', encoding="ISO-8859-1")

#df['Target'] = pd.Series(np.where(df['cNEU'] == 'y', 1, 0))
target = df['cNEU']
status = df['STATUS']

#Setting tokenizer (can replace later)
tk = nltk.tokenize.WhitespaceTokenizer()
tokenizer = tk.tokenize

#Loading LIWC dictionary
parse,categories = liwc.load_token_parser(LIWC_file)
#vector will include all categories in LIWC dictionary
#can also import custom list of specific categories to count:
#categories = load_categories("categorylist.txt")

vectorized = liwc_series_vectorizer(status,tokenizer,categories,parse)

x_train, x_test, y_train, y_test = train_test_split(vectorized, target,\
    test_size=0.2, random_state=4)

# x_train_feat_strings = convert_lines_to_feature_strings(x_train, stop_words, \
#     "", remove_stopword_bigrams=True)
#
# x_features_train, training_vectorizer = convert_text_into_features(x_train_feat_strings, \
#     stop_words, whitespace_tokenizer)
# x_test_pp = training_vectorizer.transform(x_test).toarray()
# x_train_pp = x_features_train.toarray()

mod = LogisticRegression(solver='liblinear')
mod.fit(x_train, y_train)
y_hat = mod.predict(x_test)

print(accuracy_score(y_test, y_hat))
