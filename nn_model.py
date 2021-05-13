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
stop_file = '../mallet_en_stoplist.txt'
proc_file = '../new_legis_proc_jargon_stopwords.txt'

stop_words = load_stopwords(stop_file)
proc_words = load_stopwords(proc_file)

df = pd.read_csv('../data/wcpr_mypersonality.csv', \
    encoding="ISO-8859-1")

df['Target'] = pd.Series(np.where(df['cNEU'] == 'y', 1, 0))

stopwords_dec = True

target = df['Target']
status = df[['STATUS' ,'#AUTHID']]
dist_users = df[['#AUTHID', 'Target']].drop_duplicates()

x_train, x_test, y_train, y_test = train_test_split(dist_users['#AUTHID'], dist_users['Target'],\
    test_size=0.3, random_state=4, stratify=dist_users['Target'])

train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_test, y_test], axis=1).sort_values(by=['#AUTHID'])

x_train_feat = pd.merge(status, train_data, on='#AUTHID',how='inner')
x_test_feat = pd.merge(status, test_data, on='#AUTHID',how='inner')

x_train_pp = np.array(x_train_feat['STATUS'])
x_test_pp = np.array(x_test_feat['STATUS'])

x_train_feat_strings = convert_lines_to_feature_strings(x_train_pp, stop_words, \
    proc_words, remove_stopword_bigrams=stopwords_dec)
x_test_feat_string = convert_lines_to_feature_strings(x_test_pp, stop_words, \
    proc_words, remove_stopword_bigrams=stopwords_dec)

x_features_train, training_vectorizer = convert_text_into_features(x_train_feat_strings, \
    stop_words, whitespace_tokenizer)
x_test_transformed = training_vectorizer.transform(x_test_feat_string)

x_train_mod = x_features_train.toarray()
x_test_mod = x_test_transformed.toarray()
y_train_mod = np.array(x_train_feat['Target'])


if stopwords_dec == True:
    input_size = 26811
else:
    input_size = 69353
# print(x_train_pp)

test_mod = models.Sequential([
    layers.Input(shape=(1,input_size)), 
    layers.Dense(32, activation='relu'), 
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'), 
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

metric = tf.keras.metrics.Precision()
optim = optimizers.RMSprop(lr=0.001)
test_mod.compile(optimizer='Nadam', 
                loss='binary_crossentropy', 
                metrics=[metric])

hist = test_mod.fit(x_train_mod, y_train_mod, epochs=3)
y_hat = pd.Series(np.round(test_mod.predict(x_test_mod).reshape(2825,)))
# y_hat = np.round(test_mod.predict(x_test_mod).reshape(1984,))



# # print(accuracy_score(y_test, preds))
# # print(precision_score(y_test, preds))

# # mod = LogisticRegression(solver='liblinear')
# # mod.fit(x_features_train, x_train_feat['Target'])
# # y_hat = pd.Series(mod.predict(x_test_transformed))

x_test_feat['Preds'] = y_hat

grouped_preds = x_test_feat.groupby('#AUTHID')['Preds'].sum().reset_index().sort_values(by=['#AUTHID'])
grouped_counts = x_test_feat.groupby('#AUTHID')['Preds'].count().reset_index().sort_values(by=['#AUTHID'])

joined_preds = pd.merge(grouped_preds, grouped_counts, \
    suffixes=('_preds', '_counts'),on='#AUTHID', how='inner')

pred_thresh = 0.65

joined_preds['Final_pred'] = pd.Series(\
    np.where((joined_preds['Preds_preds'] / joined_preds['Preds_counts']) >= pred_thresh, 1, 0))

print(accuracy_score(test_data['Target'], joined_preds['Final_pred']))

