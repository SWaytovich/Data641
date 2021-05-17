import pandas as pd 
import numpy as np 
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
from sklearn import tree, ensemble, svm
from sklearn.metrics import (
    accuracy_score, precision_score, 
    confusion_matrix, recall_score
)
import spacy, string 
import unicodedata
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def whitespace_tokenizer(line):
    return line.split()

def remove_accents(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', \
        'ignore').decode('utf-8', 'ignore')
    return new_text

def remove_encoding_fuckers(text):
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


stop_file = '../mallet_en_stoplist.txt'
proc_file = '../new_legis_proc_jargon_stopwords.txt'

stop_words = load_stopwords(stop_file)
proc_words = load_stopwords(proc_file)

lower_func = lambda text: text.lower()

df = pd.read_csv('../data/wcpr_mypersonality.csv', \
    encoding="ISO-8859-1")

df['Target'] = pd.Series(np.where(df['cNEU'] == 'y', 1, 0))
df['STATUS'] = df['STATUS'].map(remove_encoding_fuckers)
df['STATUS'] = df['STATUS'].map(lower_func)
df['STATUS'] = df['STATUS'].map(remove_accents)
# df['STATUS'] = df['STATUS'].map(check_punc)

df['Status_Length'] = pd.Series([len(df['STATUS'][i].split()) for i in range(len(df))])
df = df[df['Status_Length'] > 4]


target = df['Target']
status = df[['STATUS' ,'#AUTHID']]
dist_users = df[['#AUTHID', 'Target']].drop_duplicates()

x_train, x_test, y_train, y_test = train_test_split(dist_users['#AUTHID'], dist_users['Target'],\
    test_size=0.3, random_state=4, stratify=dist_users['Target'])

train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_test, y_test], axis=1).sort_values(by=['#AUTHID'])

x_train_feat = pd.merge(status, train_data, on='#AUTHID',how='inner')
x_test_feat = pd.merge(status, test_data, on='#AUTHID',how='inner')

# y_test_label = pd.concat([x_test_feat['#AUTHID'], y_test])

x_train_pp = np.array(x_train_feat['STATUS'])
x_test_pp = np.array(x_test_feat['STATUS'])

x_train_feat_strings = convert_lines_to_feature_strings(x_train_pp, stop_words, \
    proc_words, remove_stopword_bigrams=True, include_trigrams=False)
x_test_feat_string = convert_lines_to_feature_strings(x_test_pp, stop_words, \
    proc_words, remove_stopword_bigrams=True, include_trigrams=False)

x_features_train, training_vectorizer = convert_text_into_features(x_train_feat_strings, \
    stop_words, whitespace_tokenizer)
x_test_transformed = training_vectorizer.transform(x_test_feat_string)
# print(training_vectorizer.get_feature_names())
mod = LogisticRegression(solver='liblinear')
mod.fit(x_features_train, x_train_feat['Target'])
y_hat = pd.Series(mod.predict(x_test_transformed))

x_test_feat['Preds'] = y_hat

grouped_preds = x_test_feat.groupby('#AUTHID')['Preds'].sum().reset_index().sort_values(by=['#AUTHID'])
grouped_counts = x_test_feat.groupby('#AUTHID')['Preds'].count().reset_index().sort_values(by=['#AUTHID'])

joined_preds = pd.merge(grouped_preds, grouped_counts, \
    suffixes=('_preds', '_counts'),on='#AUTHID', how='inner')

pred_thresh = 0.6

joined_preds['Final_pred'] = pd.Series(\
    np.where((joined_preds['Preds_preds'] / joined_preds['Preds_counts']) >= pred_thresh, 1, 0))

print(accuracy_score(test_data['Target'], joined_preds['Final_pred']))
print(precision_score(test_data['Target'], joined_preds['Final_pred']))
print(recall_score(test_data['Target'], joined_preds['Final_pred']))