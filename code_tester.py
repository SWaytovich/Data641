import pandas as pd 
import numpy as np 
from final_proj_funcs import *
import tensorflow as tf 
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
    confusion_matrix
)
# import spacy 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def whitespace_tokenizer(line):
    return line.split()
stop_file = '../mallet_en_stoplist.txt'
proc_file = '../new_legis_proc_jargon_stopwords.txt'

stop_words = load_stopwords(stop_file)
proc_words = load_stopwords(proc_file)

df = pd.read_csv('../data/wcpr_mypersonality.csv', \
    encoding="ISO-8859-1")

df['Target'] = pd.Series(np.where(df['cNEU'] == 'y', 1, 0))
df['STATUS'] = df['STATUS'].str.replace('\x92', "'")
df['STATUS'] = df['STATUS'].str.replace('\x94', '"')
df['STATUS'] = df['STATUS'].str.replace('\x93', '"')

target = df['Target']
status = df[['STATUS' ,'#AUTHID']]
dist_users = df[['#AUTHID', 'Target']].drop_duplicates()

x_train, x_test, y_train, y_test = train_test_split(dist_users['#AUTHID'], \
    dist_users['Target'], test_size=0.3, random_state=4, \
        stratify=dist_users['Target'])

train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_test, y_test], axis=1).sort_values(by=['#AUTHID'])

x_train_feat = pd.merge(status, train_data, on='#AUTHID',how='inner')
x_test_feat = pd.merge(status, test_data, on='#AUTHID',how='inner')

x_train_pp = np.array(x_train_feat['STATUS'])
x_test_pp = np.array(x_test_feat['STATUS'])

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(x_train_feat['STATUS'])

train_seq = tokenizer.texts_to_sequences(x_train_feat['STATUS'])
test_seq = tokenizer.texts_to_sequences(x_test_feat['STATUS'])

train_lens = np.array([len(i) for i in train_seq]).max()
train_data_mod = pad_sequences(train_seq, maxlen=train_lens)
test_data_mod = pad_sequences(test_seq, maxlen=train_lens)

train_labels, test_labels = np.array(x_train_feat['Target']),  \
    np.array(x_test_feat['Target'])

embedding_dim = 70
RNN_model = models.Sequential([
    layers.Embedding(20000, embedding_dim), 
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)), 
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation='relu'), 
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
rms = optimizers.RMSprop(lr=1e-4)
RNN_model.compile(optimizer=optimizers.Adam(1e-4),
    loss='binary_crossentropy', metrics=['acc'])
RNN_model.fit(train_data_mod, train_labels, epochs=10, steps_per_epoch=100)
print(RNN_model.predict(test_data_mod).shape)
check_preds = RNN_model.predict(test_data_mod)
y_hat = pd.Series(np.round(RNN_model.predict(test_data_mod).reshape(2825,)))
print(accuracy_score(test_labels, y_hat))
x_test_feat['Preds'] = y_hat

grouped_preds = x_test_feat.groupby('#AUTHID')['Preds'].sum().reset_index()
grouped_counts = x_test_feat.groupby('#AUTHID')['Preds'].count().reset_index()

joined_preds = pd.merge(grouped_preds, grouped_counts, \
    suffixes=('_preds', '_counts'),on='#AUTHID', how='inner')

pred_thresh = 0.7
joined_preds['Final_pred'] = pd.Series(\
    np.where((joined_preds['Preds_preds'] / joined_preds['Preds_counts']) >= pred_thresh, 1, 0))

print(accuracy_score(test_data['Target'], joined_preds['Final_pred']))
print(precision_score(test_data['Target'], joined_preds['Final_pred']))

# nlp = English(parser=False)
# spacy_an = nlp(x_train_pp[1])
# spacy_tokens = [token.orth_ for token in spacy_an]
# print(spacy_tokens)
# x_train_feat_strings = convert_lines_to_feature_strings(x_train_pp, stop_words, \
#     proc_words, remove_stopword_bigrams=True)
# x_test_feat_string = convert_lines_to_feature_strings(x_test_pp, stop_words, \
#     proc_words, remove_stopword_bigrams=True)

# x_features_train, training_vectorizer = convert_text_into_features(x_train_feat_strings, \
#     stop_words, whitespace_tokenizer)
# x_test_transformed = training_vectorizer.transform(x_test_feat_string)

# print(training_vectorizer.get_feature_names())
