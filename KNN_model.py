'''
To run this code and the other code like it that were submitted, the final_proj_funcs program will need to be in the same folder
and then this code just needs to be exceuted. 
The data file location can be edited below when the funciton is called 
'''

import pandas as pd 
import numpy as np 

# Important Functions used for the Trigrams feature computation
from final_proj_funcs import *
# import tensorflow as tf 
# from tensorflow.keras import (
#     models, losses, optimizers, 
#     layers, preprocessing, 
# )
from sklearn.model_selection import (
    train_test_split, KFold, 
    StratifiedKFold, cross_val_score
)
from sklearn.linear_model import LogisticRegression 
from sklearn import tree, ensemble, svm, naive_bayes, neighbors, gaussian_process, cluster
from sklearn.metrics import (
    accuracy_score, precision_score, 
    confusion_matrix, f1_score, recall_score
)
import spacy, re, nltk 
import unicodedata, string
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Functions for cleaning up the text 
def remove_numbers(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pattern, '', text)

def get_stem(text):
    stemmer = nltk.porter.PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

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

def check_punc(text):
    punc_indices = []
    for punc in string.punctuation:
        if punc in text:
            text = text.replace(punc, '')
    return text 

# File names for the stopwords 
stop_file = '../mallet_en_stoplist.txt'
proc_file = '../new_legis_proc_jargon_stopwords.txt'

stop_words = load_stopwords(stop_file)
proc_words = load_stopwords(proc_file)

lower_func = lambda text: text.lower()

df = pd.read_csv('../data/wcpr_mypersonality.csv', \
    encoding="ISO-8859-1")

# Classifier
knn = neighbors.KNeighborsClassifier(n_neighbors=17, weights='distance')
# mod = LogisticRegression(solver='liblinear')
# knn_cluster = cluster.KMeans(n_clusters=2)
# nb_clf = gaussian_process.GaussianProcessClassifier()

# Creating a numerical Binary target column and 
# normalizing the statuses 
df['Target'] = pd.Series(np.where(df['cNEU'] == 'y', 1, 0))
df['STATUS'] = df['STATUS'].map(remove_encoding)
df['STATUS'] = df['STATUS'].map(lower_func)
df['STATUS'] = df['STATUS'].map(remove_accents)
df['STATUS'] = df['STATUS'].map(check_punc)
df['STATUS'] = df['STATUS'].map(remove_numbers)
df['STATUS'] = df['STATUS'].map(get_stem)

# Limiting the posts to greater than 4 words 
df['Status_Length'] = pd.Series([len(df['STATUS'][i].split()) for i in range(len(df))])
df = df[df['Status_Length'] > 4]

# Getting distinct users and targets 
target = df['Target']
status = df[['STATUS' ,'#AUTHID']]
dist_users = df[['#AUTHID', 'Target']].drop_duplicates().reset_index()

# Combining the posts of a single user into one 
feat_strings = []
for indx in range(len(dist_users)):
    if len(df[df['#AUTHID'] == dist_users.iloc[indx]['#AUTHID']]) > 1:
        temp_df = df[df['#AUTHID'] == dist_users.iloc[indx]['#AUTHID']]
        feat_strings.append(" ".join(temp_df.iloc[i]['STATUS'] for i in range(len(temp_df))))
    else:
        feat_strings.append(df[df['#AUTHID'] == dist_users.iloc[indx]['#AUTHID']].iloc[0]['STATUS'])
dist_users['Full_Posts'] = pd.Series(feat_strings)

kf_splits = StratifiedKFold(n_splits=5)

# Boolean to either run with Train test split or KFold
# Change the booelan below to use Kfold or not 
use_kfold = True 

if use_kfold == True:
        
    acc_score = []
    prec_score = []
    f1_score = []

    for train_indx, test_indx in kf_splits.split(dist_users['Full_Posts'], dist_users['Target']):
        x_train , x_test = dist_users.iloc[train_indx,:]['Full_Posts'],dist_users.iloc[test_indx,:]['Full_Posts']
        y_train , y_test = dist_users.iloc[train_indx]['Target'] , dist_users.iloc[test_indx]['Target']
        # print(train_indx)
        # print(test_indx)
        x_train_pp = np.array(x_train)
        x_test_pp = np.array(x_test)
        x_train_feat_strings = convert_lines_to_feature_strings(x_train_pp, stop_words, \
            proc_words, remove_stopword_bigrams=False, include_trigrams=True)
        x_test_feat_string = convert_lines_to_feature_strings(x_test_pp, stop_words, \
            proc_words, remove_stopword_bigrams=False, include_trigrams=True)

        x_features_train, training_vectorizer = convert_text_into_features(x_train_feat_strings, \
            stop_words, whitespace_tokenizer)
        x_test_transformed = training_vectorizer.transform(x_test_feat_string)

        knn.fit(x_features_train, y_train)
        print('got here')
        preds = knn.predict(x_test_transformed)
        acc_score.append(accuracy_score(y_test, preds))
        prec_score.append(precision_score(y_test, preds))
        f1_score.append(recall_score(y_test, preds))
        # print(precision_score(y_test, preds))
        # print(accuracy_score(y_test, preds))
        # # print(f1_score(y_test, preds))
        # print(recall_score(y_test, preds))
        # print('\n')
    print(np.mean(acc_score))
    print(np.mean(prec_score))
    print(np.mean(f1_score))
    
else: 
    x_train, x_test, y_train, y_test = train_test_split(dist_users[['#AUTHID', 'Full_Posts']], dist_users['Target'],\
        test_size=0.33, random_state=4, stratify=dist_users['Target'])

    x_train_pp = np.array(x_train['Full_Posts'])
    x_test_pp = np.array(x_test['Full_Posts'])

    x_train_feat_strings = convert_lines_to_feature_strings(x_train_pp, stop_words, \
        proc_words, remove_stopword_bigrams=False, include_trigrams=True)
    x_test_feat_string = convert_lines_to_feature_strings(x_test_pp, stop_words, \
        proc_words, remove_stopword_bigrams=False, include_trigrams=True)

    x_features_train, training_vectorizer = convert_text_into_features(x_train_feat_strings, \
        stop_words, whitespace_tokenizer)
    x_test_transformed = training_vectorizer.transform(x_test_feat_string)

    
    knn.fit(x_features_train, y_train)
    preds = knn.predict(x_test_transformed)

    print(precision_score(y_test, preds))
    print(accuracy_score(y_test, preds))
    print(f1_score(y_test, preds))
    print(recall_score(y_test, preds))