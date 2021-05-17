import pandas as pd
import numpy as np
from final_proj_funcs import * #laoding auxiliary functions
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
    recall_score,confusion_matrix,f1_score
)
import unicodedata
import spacy
import os

import nltk,re
from liwc_funcs import * #loading auxiliary functions
import liwc

#Use all categories of LIWC, or 10 most highly correlated?
all_LIWC = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
random_seed = 13

#Defining
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

df = pd.read_csv('wcpr_mypersonality.csv',encoding="ISO-8859-1")

#df['Target'] = df['cNEU']
df['Target'] = pd.Series(np.where(df['cNEU'] == 'y', 1, 0))

target = df['Target']
status = df[['STATUS' ,'#AUTHID']]
dist_users = df[['#AUTHID', 'Target']].drop_duplicates()

newdf = df.groupby('#AUTHID').agg({'STATUS': lambda x: ' '.join(x), 'NETWORKSIZE': 'mean', 'BETWEENNESS':'mean',
       'NBETWEENNESS':'mean', 'DENSITY':'mean', 'BROKERAGE':'mean', 'NBROKERAGE':'mean', 'TRANSITIVITY':'mean',
       'Target':'mean'})

postcounter = df.groupby('#AUTHID').agg(PostCt =('TRANSITIVITY','count'))
newdf['PostCt'] = postcounter['PostCt'].to_numpy()

#cleaning status data
newdf['STATUS'] = newdf['STATUS'].apply(remove_accents).apply(remove_encoding)

#ADDING OTHER VARIABLES TO REGRESSION
newx = newdf[['STATUS','NETWORKSIZE', 'BETWEENNESS', 'NBETWEENNESS', 'DENSITY',
        'BROKERAGE', 'NBROKERAGE', 'TRANSITIVITY','PostCt']]

kf_splits =  StratifiedKFold(n_splits=5,shuffle=True,random_state =random_seed)

###########################################################################
#Loading LIWC and Checking Correlations
#Finding correlations:
LIWC_file = 'myliwc.dic'
tk = nltk.tokenize.WhitespaceTokenizer()
tokenizer = tk.tokenize
#
#Loading LIWC dictionary
parse,categories = liwc.load_token_parser(LIWC_file)
#by default, vector will include all categories in LIWC dictionary

newdf_LIWC = liwc_series_vectorizer(newdf["STATUS"],tokenizer,categories,parse)
newdf_LIWC["Target"] = newdf['Target'].to_numpy()
cordf = newdf_LIWC.corr()
targ_cor = cordf['Target']
print("Correlations with Target(Neurotic):")
print(targ_cor.sort_values(ascending = False)[1:6])
print(targ_cor.sort_values(ascending = True)[0:5])

#If all_LIWC = False, then vector used for regressions will only include
# top 5 postive correlated and top 5 negative correlated with cNEU/target
if all_LIWC==False:
    categories = ['anger','excl','swear','time','negemo','posemo','preps','filler','social','future']
    #categories = ['anger','negemo','death','excl','body','preps','achieve','social','posemo','filler']

######################

acc_scores = []
prec_scores = []
recall_scores = []
f1_scores = []

train_acc = []
train_prec = []
train_recall = []
train_f1 = []

for train_idx,test_idx in kf_splits.split(newx,newdf['Target']):
    x_train , x_test = newdf.iloc[train_idx],newdf.iloc[test_idx]
    y_train , y_test = newdf.iloc[train_idx]['Target'] , newdf.iloc[test_idx]['Target']

    x_train_LIWC = liwc_series_vectorizer(x_train["STATUS"],tokenizer,categories,parse)
    x_test_LIWC = liwc_series_vectorizer(x_test["STATUS"],tokenizer,categories,parse)
    #
    #Dividing word ct by post ct:
    wc_tr = x_train_LIWC["WordCount"].to_numpy()
    pc_tr = x_train["PostCt"].to_numpy()
    avgw_tr = wc_tr/pc_tr
    #
    x_train_LIWC["WordCount"] = pd.Series(avgw_tr)
    x_train_LIWC["PostCt"] = pc_tr
    #
    wc_ts = x_test_LIWC["WordCount"].to_numpy()
    pc_ts = x_test["PostCt"].to_numpy()
    avgw_ts = wc_ts/pc_ts
    #
    x_test_LIWC["WordCount"] = pd.Series(avgw_ts)
    x_test_LIWC["PostCt"] = pc_ts
    #
    mod_LIWC = LogisticRegression(solver='liblinear',class_weight='balanced')
    #mod_LIWC = LogisticRegression(solver='liblinear')
    mod_LIWC.fit(x_train_LIWC, y_train)
    y_hat_LIWC = pd.Series(mod_LIWC.predict(x_test_LIWC))
    #
    acc_scores.append(accuracy_score(y_test, y_hat_LIWC))
    prec_scores.append(precision_score(y_test, y_hat_LIWC))
    recall_scores.append(recall_score(y_test, y_hat_LIWC))
    f1_scores.append(f1_score(y_test,y_hat_LIWC))

    y_train_hat = pd.Series(mod_LIWC.predict(x_train_LIWC))
    train_acc.append(accuracy_score(y_train, y_train_hat))
    train_prec.append(precision_score(y_train, y_train_hat))
    train_recall.append(recall_score(y_train, y_train_hat))
    train_f1.append(f1_score(y_train,y_train_hat))

print("\nOn Training Set:")
print("Accuracy: {} Precision: {}\nRecall: {} F1 Score: {}".format(np.mean(train_acc),np.mean(train_prec),np.mean(train_recall),np.mean(train_f1)))
print("On Test Set:")
print("Accuracy: {} Precision: {}\nRecall: {} F1 Score: {}".format(np.mean(acc_scores),np.mean(prec_scores),np.mean(recall_scores),np.mean(f1_scores)))
