
# System imports
import os, sys, codecs
import argparse, json, gzip, re
import string
from collections import Counter

# Helper imports
# - Verbose info for debugging
from traceback_with_variables import activate_by_import
# - Progress bars
from tqdm import tqdm # Runtime progress bar

# Machine learning imports
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

# NLP imports
from spacy.lang.en import English
import spacy 


def load_stopwords(filename):
    stop_file = open(filename, 'r')
    lines = stop_file.readlines()
    stopwords = [i.strip() for i in lines] # ASSIGNMENT: replace this with your code
    return set(stopwords)


def normalize_tokens(tokenlist):
    # Input: list of tokens as strings,  e.g. ['I', ' ', 'saw', ' ', '@psresnik', ' ', 'on', ' ','Twitter']
    # Output: list of tokens where
    #   - All tokens are lowercased
    #   - All tokens starting with a whitespace character have been filtered out
    #   - All handles (tokens starting with @) have been filtered out
    #   - Any underscores have been replaced with + (since we use _ as a special character in bigrams)
    normalized_tokens = [token.lower().replace('_','+') for token in tokenlist   # lowercase, _ => +
                             if re.search('[^\s]', token) is not None            # ignore whitespace tokens
                             and not token.startswith("@")                       # ignore  handles
                        ]
    return normalized_tokens        


# Take a list of string tokens and return all ngrams of length n,
# representing each ngram as a list of  tokens.
# E.g. ngrams(['the','quick','brown','fox'], 2)
# returns [['the','quick'], ['quick','brown'], ['brown','fox']]
# Note that this should work for any n, not just unigrams and bigrams
def ngrams(tokens, n):
    # Returns all ngrams of size n in sentence, where an ngram is itself a list of tokens
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]

def filter_punctuation_bigrams(ngrams):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams
    # Removes ngrams like ['today','.'] where either token is a punctuation character
    # Returns list with the items that were not removed
    punct = string.punctuation
    return [ngram   for ngram in ngrams   if ngram[0] not in punct and ngram[1] not in punct]

def filter_stopword_bigrams(ngrams, stopwords):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams, stopwords is a set of words like 'the'
    # Removes ngrams like ['in','the'] and ['senator','from'] where either word is a stopword
    # Returns list with the items that were not removed
    result = [ngram   for ngram in ngrams   if ngram[0] not in stopwords and ngram[1] not in stopwords]
    return result

def read_and_clean_lines(infile):
    print("\nReading and cleaning text from {}".format(infile))
    lines = []
    parties = []
    with gzip.open(infile, 'rt') as f:
        for line in tqdm(f):
            line_json = json.loads(line)
            if line_json['chamber'].lower() == 'senate':
                lines.append(line_json['party'] + \
                    '\t' + line_json['text'].replace('\n', ''))
                parties.append(line_json['party'])
    # TO DO: Your code goes here
    print("Read {} documents".format(len(lines)))
    print("Read {} labels".format(len(parties)))
    return lines, parties

# This function should return those four values
def split_training_set(lines, labels, test_size=0.3, random_seed=42):
    # ss_split = StratifiedShuffleSplit(n_splits=1, random_state=random_seed, test_size=test_size)
    # train_data, test_data = ss_split.split(lines, labels)
    # X_train, X_test = lines[train_data], lines[test_data]
    # y_train, y_test = labels[train_data], labels[test_data]
    # TO DO: replace this line with a call to train_test_split
    X_train, X_test, y_train, y_test = train_test_split(lines, labels, test_size=test_size, \
        random_state=random_seed, stratify=labels)
    print("Training set label counts: {}".format(Counter(y_train)))
    print("Test set     label counts: {}".format(Counter(y_test)))
    return X_train, X_test, y_train, y_test

# Converting text into features.
# Inputs:
#    X - a sequence of raw text strings to be processed
#    analyzefn - either built-in (see CountVectorizer documentation), or a function we provide from strings to feature-lists
#
#    Arguments used by the words analyzer
#      stopwords - set of stopwords (used by "word" analyzer")
#      lowercase - true if normalizing by lowercasing
#      ngram_range - (N,M) for using ngrams of sizes N up to M as features, e.g. (1,2) for unigrams and bigrams
#
#  Outputs:
#     X_features - corresponding feature vector for each raw text item in X
#     training_vectorizer - vectorizer object that can now be applied to some new X', e.g. containing test texts
#    
# You can find documentation at https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# and there's a nice, readable discussion at https://medium.com/swlh/understanding-count-vectorizer-5dd71530c1b
#
def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1,2)):
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer

# Input:
#    lines     - a raw text corpus, where each element in the list is a string
#    stopwords - a set of strings that are stopwords
#    remove_stopword_bigrams = True or False
#
# Output:  a corresponding list converting the raw strings to space-separated features
#
# The features extracted should include non-stopword, non-punctuation unigrams,
# plus the bigram features that were counted in collect_bigram_counts from the previous assignment
# represented as underscore_separated tokens.
# Example:
#   Input:  ["This is Remy's dinner.",
#            "Remy will eat it."]
#   Output: ["remy 's dinner remy_'s 's_dinner",
#            "remy eat"]
def convert_lines_to_feature_strings(lines, stopwords, proc_words,remove_stopword_bigrams=True, include_trigrams=True):

    print(" Converting from raw text to unigram and bigram features")
    if remove_stopword_bigrams:
        print(" Includes filtering stopword bigrams")
        
    print(" Initializing")
    nlp          = English(parser=False)
    # nlp = spacy.load('en_core_web_sm')
    all_features = []
    print(" Iterating through documents extracting unigram and bigram features")
    for line in tqdm(lines):
        
        # Get spacy tokenization and normalize the tokens
        spacy_analysis    = nlp(line)
        spacy_tokens      = [token.orth_ for token in spacy_analysis]
        # spacy_tokens      = [token.lemma_ for token in spacy_analysis]
        # print(spacy_tokens)
        normalized_tokens = normalize_tokens(spacy_tokens)

        # Collect unigram tokens as features
        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        unigrams          = [token   for token in normalized_tokens
                                if token not in stopwords and token not in string.punctuation]
                                # if token not in stopwords and token not in string.punctuation and token not in proc_words]

        # Collect string bigram tokens as features
        bigrams = []
        bigram_tokens     = ["_".join(bigram) for bigram in bigrams]
        bigrams           = ngrams(normalized_tokens, 2) 
        bigrams           = filter_punctuation_bigrams(bigrams)
        if remove_stopword_bigrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)
            # bigrams = filter_stopword_bigrams(bigrams, proc_words)
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]

        # Collect string trigram tokens as features 
        # trigrams = []
        # trigram_tokens     = ["_".join(trigram) for trigram in trigrams]
        trigrams           = ngrams(normalized_tokens, 3) 
        trigrams           = filter_punctuation_bigrams(trigrams)
        if remove_stopword_bigrams:
            trigrams = filter_stopword_bigrams(trigrams, stopwords)
            # trigrams = filter_stopword_bigrams(trigrams, proc_words)
        trigram_tokens = ["_".join(trigram) for trigram in trigrams]
        # Conjoin the feature lists and turn into a space-separated string of features.
        # E.g. if unigrams is ['coffee', 'cup'] and bigrams is ['coffee_cup', 'white_house']
        # then feature_string should be 'coffee cup coffee_cup white_house'
        if include_trigrams == True:
        # TO DO: replace this line with your code
            feature_string = " ".join(token for token in unigrams) + " " \
                + " ".join(bigram for bigram in bigram_tokens) + " ".join(trigram for trigram in trigram_tokens)
        else:
            feature_string = " ".join(token for token in unigrams) + " " \
            + " ".join(bigram for bigram in bigram_tokens) 
        # Add this feature string to the output
        all_features.append(feature_string)


    # print(" Feature string for first document: '{}'".format(all_features[0]))
        
    return all_features
