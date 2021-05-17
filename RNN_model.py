'''
To run this code and the other code like it that were submitted, the final_proj_funcs program will need to be in the same folder
and then this code just needs to be exceuted. 
The data file location can be edited below when the funciton is called 
'''

import pandas as pd 
import numpy as np 
import collections, string 
from final_proj_funcs import *
import tensorflow as tf 
import unicodedata
from tensorflow.keras import (
    models, losses, optimizers, 
    layers, preprocessing, 
)
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import (
    train_test_split, KFold, 
    StratifiedKFold, cross_val_score
)
from sklearn.linear_model import LogisticRegression 
from sklearn import tree, ensemble, svm
from sklearn.metrics import (
    accuracy_score, precision_score, 
    confusion_matrix, f1_score
)
import spacy
import os, re, nltk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Functions for cleaning and processing the text 
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


stop_file = '../mallet_en_stoplist.txt'
proc_file = '../new_legis_proc_jargon_stopwords.txt'

stop_words = load_stopwords(stop_file)
proc_words = load_stopwords(proc_file)

# Function for running the model
# IS called below so this script just needs to be executed to run the model and view the metrics 
def model_run(file_path, stopwords_dec, test_size, pred_thresh):
    df = pd.read_csv(file_path, \
        encoding="ISO-8859-1")
    token_pipe = spacy.lang.en.English(parser=False)
    lower_func = lambda text: text.lower()

    df['Target'] = pd.Series(np.where(df['cNEU'] == 'y', 1, 0))
    df['STATUS'] = df['STATUS'].map(remove_encoding)
    df['STATUS'] = df['STATUS'].map(lower_func)
    df['STATUS'] = df['STATUS'].map(remove_accents)
    df['STATUS'] = df['STATUS'].map(check_punc)
    df['STATUS'] = df['STATUS'].map(remove_numbers)
    df['STATUS'] = df['STATUS'].map(get_stem)


    df['Status_Length'] = pd.Series([len(df['STATUS'][i].split()) for i in range(len(df))])
    df = df[df['Status_Length'] > 4]
    # print(df['Status_Length'].mean())
    target = df['Target']
    status = df[['STATUS' ,'#AUTHID']]
    dist_users = df[['#AUTHID', 'Target']].drop_duplicates()

    x_train, x_test, y_train, y_test = train_test_split(dist_users['#AUTHID'], \
        dist_users['Target'], test_size=test_size, random_state=4, \
            stratify=dist_users['Target'])

    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1).sort_values(by=['#AUTHID'])

    x_train_feat = pd.merge(status, train_data, on='#AUTHID',how='inner')
    x_test_feat = pd.merge(status, test_data, on='#AUTHID',how='inner')
    train_labels, test_labels = np.array(x_train_feat['Target']),  \
        np.array(x_test_feat['Target'])

    # Alternate tokenizer other than the Spacy one used in other programs 
    tokenizer = Tokenizer(num_words=40000)
    tokenizer.fit_on_texts(x_train_feat['STATUS'])

    train_seq = tokenizer.texts_to_sequences(x_train_feat['STATUS'])
    test_seq = tokenizer.texts_to_sequences(x_test_feat['STATUS'])

    # Pad the sequences so that the posts are equal length 
    train_lens = np.array([len(i) for i in train_seq]).max()
    train_data_mod = pad_sequences(train_seq, maxlen=train_lens)
    test_data_mod = pad_sequences(test_seq, maxlen=train_lens)

    # x_train_pp = np.array(x_train_feat['STATUS'])
    # x_test_pp = np.array(x_test_feat['STATUS'])

    # x_train_feat_strings = convert_lines_to_feature_strings(x_train_pp, stop_words, \
    #     proc_words, remove_stopword_bigrams=stopwords_dec, include_trigrams=False)
    # x_test_feat_string = convert_lines_to_feature_strings(x_test_pp, stop_words, \
    #     proc_words, remove_stopword_bigrams=stopwords_dec, include_trigrams=False)

    # x_features_train, training_vectorizer = convert_text_into_features(x_train_feat_strings, \
    #     stop_words, whitespace_tokenizer)
    # x_test_transformed = training_vectorizer.transform(x_test_feat_string)

    # x_train_mod = x_features_train.toarray().reshape(x_features_train.shape[0], 1, x_features_train.shape[1])
    # x_test_mod = x_test_transformed.toarray().reshape(x_test_transformed.shape[0], 1, x_test_transformed.shape[1])
    # y_train_mod = np.array(x_train_feat['Target'])
    # print(x_features_train.shape)

    # RNN model 
    embedding_dim = 70
    RNN_model = models.Sequential([
        layers.Embedding(40000, embedding_dim), 
        layers.Bidirectional(layers.LSTM(124, return_sequences=True)), 
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'), 
        # layers.Dropout(0.2),
        # layers.BatchNormalization(), 
        layers.Dense(1, activation='sigmoid')
    ])

    # Callback function and loss and optimizers 
    callback = tf.keras.callbacks.EarlyStopping(monitor='precision', patience=3)
    prec = tf.keras.metrics.Precision()
    rms = optimizers.RMSprop(lr=1e-4)
    adam = optimizers.Adam(1e-4)

    RNN_model.compile(optimizer='adam',
        loss='binary_crossentropy', metrics=[prec])
    RNN_model.fit(train_data_mod, train_labels, epochs=10, \
        steps_per_epoch=100, callbacks=[callback])
    out_shape = RNN_model.predict(test_data_mod)

    # Predictions 
    y_hat = pd.Series(np.round(RNN_model.predict(test_data_mod).reshape(out_shape.shape[0],)))
    print(accuracy_score(test_labels, y_hat))
    x_test_feat['Preds'] = y_hat

    # Grouping by users 
    grouped_preds = x_test_feat.groupby('#AUTHID')['Preds'].sum().reset_index()
    grouped_counts = x_test_feat.groupby('#AUTHID')['Preds'].count().reset_index()

    joined_preds = pd.merge(grouped_preds, grouped_counts, \
        suffixes=('_preds', '_counts'),on='#AUTHID', how='inner')

    joined_preds['Final_pred'] = pd.Series(\
        np.where((joined_preds['Preds_preds'] / joined_preds['Preds_counts']) >= pred_thresh, 1, 0))
    print(joined_preds)
    print(test_data[['#AUTHID', 'Target']])

    print(accuracy_score(test_data['Target'], joined_preds['Final_pred']))
    print(precision_score(test_data['Target'], joined_preds['Final_pred']))
    print(f1_score(test_data['Target'], joined_preds['Final_pred']))
    # return joined_preds, test_data

model_run('../data/wcpr_mypersonality.csv', False, 0.3, 0.6)

    