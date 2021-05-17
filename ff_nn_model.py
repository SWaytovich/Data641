'''
To run this code and the other code like it that were submitted, the final_proj_funcs program will need to be in the same folder
and then this code just needs to be exceuted. 
The data file location can be edited below when the funciton is called 
'''

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
    confusion_matrix, recall_score
)
import spacy, nltk 
import unicodedata
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def whitespace_tokenizer(line):
    return line.split()
def get_stem(text):
    stemmer = nltk.porter.PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def remove_accents(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', \
        'ignore').decode('utf-8', 'ignore')
    return new_text

def remove_encoding_fuckers(text):
    new_text = text.replace('\x92', "'")
    new_text = new_text.replace('\x94', '"')
    new_text = new_text.replace('\x93', '"')
    return new_text

stop_file = '../mallet_en_stoplist.txt'
proc_file = '../new_legis_proc_jargon_stopwords.txt'

stop_words = load_stopwords(stop_file)
proc_words = load_stopwords(proc_file)

def model_run(file_path, stopwords_dec, test_size, pred_thresh):

    alt_df = pd.read_csv('../data/wcpr_essays.csv', encoding="ISO-8859-1")
    alt_df['Target'] = pd.Series(np.where(alt_df['cNEU'] == 'y', 1, 0))
    alt_df['TEXT'] = alt_df['TEXT'].str.replace('\x92', "'")
    alt_df['TEXT'] = alt_df['TEXT'].str.replace('\x94', '"')
    alt_df['TEXT'] = alt_df['TEXT'].str.replace('\x93', '"')
    alt_df['Status_Length'] = pd.Series([len(alt_df['TEXT'][i].split()) for i in range(len(alt_df))])
    alt_df = alt_df[alt_df['Status_Length'] > 4]
    alt_target = alt_df['Target']
    alt_status = alt_df[['TEXT' ,'#AUTHID']]


    valid_feat_strings = convert_lines_to_feature_strings(np.array(alt_status['TEXT']), stop_words, \
        proc_words, remove_stopword_bigrams=stopwords_dec, include_trigrams=True)

    df = pd.read_csv(file_path, \
        encoding="ISO-8859-1")

    df['Target'] = pd.Series(np.where(df['cNEU'] == 'y', 1, 0))
    df['STATUS'] = df['STATUS'].str.replace('\x92', "'")
    df['STATUS'] = df['STATUS'].str.replace('\x94', '"')
    df['STATUS'] = df['STATUS'].str.replace('\x93', '"')
    # df['STATUS'] = df['STATUS'].map(remove_accents)
    # df['STATUS'] = df['STATUS'].map(check_punc)
    # df['STATUS'] = df['STATUS'].map(remove_numbers)
    df['STATUS'] = df['STATUS'].map(get_stem)
    df['Status_Length'] = pd.Series([len(df['STATUS'][i].split()) for i in range(len(df))])
    df = df[df['Status_Length'] > 4]

    target = df['Target']
    status = df[['STATUS' ,'#AUTHID']]
    dist_users = df[['#AUTHID', 'Target']].drop_duplicates()

    x_train, x_test, y_train, y_test = train_test_split(dist_users['#AUTHID'], dist_users['Target'],\
        test_size=test_size, random_state=4, stratify=dist_users['Target'])

    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1).sort_values(by=['#AUTHID'])

    x_train_feat = pd.merge(status, train_data, on='#AUTHID',how='inner')
    x_test_feat = pd.merge(status, test_data, on='#AUTHID',how='inner')

    x_train_pp = np.array(x_train_feat['STATUS'])
    x_test_pp = np.array(x_test_feat['STATUS'])

    x_train_feat_strings = convert_lines_to_feature_strings(x_train_pp, stop_words, \
        proc_words, remove_stopword_bigrams=stopwords_dec, include_trigrams=True)
    x_test_feat_string = convert_lines_to_feature_strings(x_test_pp, stop_words, \
        proc_words, remove_stopword_bigrams=stopwords_dec, include_trigrams=True)

    x_features_train, training_vectorizer = convert_text_into_features(x_train_feat_strings, \
        stop_words, whitespace_tokenizer)
    x_test_transformed = training_vectorizer.transform(x_test_feat_string)
    valid_transformed = training_vectorizer.transform(valid_feat_strings).toarray()
    valid_y = np.array(alt_target)

    x_train_mod = x_features_train.toarray()
    x_test_mod = x_test_transformed.toarray()
    y_train_mod = np.array(x_train_feat['Target'])


    if stopwords_dec == True:
        input_size = 26718
    else:
        input_size = 62690

    test_mod = models.Sequential([
        layers.Input(shape=(x_train_mod.shape[1],)), 
        layers.Dense(32, activation='relu'), 
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    metric = tf.keras.metrics.Precision()
    optim = optimizers.RMSprop(lr=0.0001)
    test_mod.compile(optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=[metric])
    callback = tf.keras.callbacks.EarlyStopping(monitor='precision', patience=3)
    hist = test_mod.fit(x_train_mod, y_train_mod, epochs=10, steps_per_epoch=100, callbacks=[callback])
    out_shape = test_mod.predict(x_test_mod).shape
    y_hat = pd.Series(np.round(test_mod.predict(x_test_mod).reshape(out_shape[0],)))
    # y_hat = test_mod.predict(x_test_mod).reshape(out_shape[0],)
    # y_hat = pd.Series(np.round(test_mod.predict(x_test_mod).reshape(2825,)))

    x_test_feat['Preds'] = y_hat

    grouped_preds = x_test_feat.groupby('#AUTHID')['Preds'].sum().reset_index().sort_values(by=['#AUTHID'])
    grouped_counts = x_test_feat.groupby('#AUTHID')['Preds'].count().reset_index().sort_values(by=['#AUTHID'])

    joined_preds = pd.merge(grouped_preds, grouped_counts, \
        suffixes=('_preds', '_counts'),on='#AUTHID', how='inner')
    # grouped_preds['Final_pred'] = pd.Series(np.round(y_hat))
    joined_preds['Final_pred'] = pd.Series(\
        np.where((joined_preds['Preds_preds'] / joined_preds['Preds_counts']) >= pred_thresh, 1, 0))

    valid_preds = np.round(test_mod.predict(valid_transformed).reshape(2467,))
    print(accuracy_score(test_data['Target'], joined_preds['Final_pred']))
    print(precision_score(test_data['Target'], joined_preds['Final_pred']))
    print(recall_score(test_data['Target'], joined_preds['Final_pred']))
    # print(valid_preds)
    print(accuracy_score(valid_y, valid_preds))
    print(precision_score(valid_y, valid_preds))
    print(recall_score(valid_y, valid_preds))
    # return x_train_mod

model_run('../data/wcpr_mypersonality.csv', False, 0.3, 0.6)