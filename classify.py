import pandas as pd
import numpy as np

import pickle
import os
import argparse
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.text_processing import process_text

TRAIN_POS_PATH = 'aclImdb/train/pos/'
TRAIN_NEG_PATH = 'aclImdb/train/neg/'

MOD_PATH = 'model/'
DATA_PATH = 'data/'
CSV_PATH = 'csv/'

MOD_FILE = 'model.pkl'
VEC_FILE = 'vectorizer.pkl'
TRAIN_FILE = 'train.pkl'

SEED = 42

mod_dir = set(os.listdir(MOD_PATH))

def read_files():
    if not 'train.csv' in set(os.listdir(CSV_PATH)):
        train = []
        
        print('reading files...')
        
        for file_ in tqdm(os.listdir(TRAIN_POS_PATH)):
            with open(TRAIN_POS_PATH + file_, 'r') as f:
                train.append([f.read(), file_[:-4].split('_')[1]])

        for file_ in tqdm(os.listdir(TRAIN_NEG_PATH)):
            with open(TRAIN_NEG_PATH + file_, 'r') as f:
                train.append([f.read(), file_[:-4].split('_')[1]])
            
        train_df = pd.DataFrame(train, columns=['Review', 'Rating'])
        train_df.to_csv('csv/train.csv', index=False)

    else:
        train_df = pd.read_csv(CSV_PATH + 'train.csv')

    return train_df

def preprocess():

    if not TRAIN_FILE in set(os.listdir(DATA_PATH)):
        train_df = read_files()

        print('preprocessing text...')

        train_df['review_parse'] = process_text(train_df['Review'])
        train_df[['review_parse', 'Rating']].to_pickle('data/train.pkl')
    else:
        train_df = pd.read_pickle(DATA_PATH + TRAIN_FILE)
        
    return train_df
    
def training():

    train_df = preprocess()
    vectorizer = TfidfVectorizer(max_df=0.95)

    X_train = vectorizer.fit_transform(train_df['review_parse'])
    y_train = train_df['Rating']

    print('training model...')

    model = OneVsOneClassifier(LogisticRegression(C=2, penalty='l2', solver='liblinear', random_state=SEED))
    model.fit(X_train, y_train)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer

def get_model():
    if not MOD_FILE in mod_dir or not VEC_FILE in mod_dir:
        model, vectorizer = training()
    else:
        with open(MOD_PATH + MOD_FILE, 'rb') as m, open(MOD_PATH + VEC_FILE, 'rb') as v:
            model = pickle.load(m)
            vectorizer = pickle.load(v)
    return model, vectorizer

def get_review_filename():
    parser = argparse.ArgumentParser()
    parser.add_argument("review_filename", help="path to a file with a movie review")
    args = parser.parse_args()
    return args.review_filename

def read_review_file(filename):
    try:
        with open(filename, 'r') as f:
            text = ''.join(list(filter(lambda s: s, map(str.strip, f))))
    except EnvironmentError:
        raise RuntimeError('File {} cannot be read/found'.format(filename))
    return text
    
def classify_review(rev_text):
    proc_rev_text = process_text([rev_text])
    
    model, vectorizer = get_model()
    text_feat = vectorizer.transform(proc_rev_text)
    
    rating = model.predict(text_feat)
    
    return rating[0], int(rating) >= 7
    
if __name__ == '__main__':
    rev_file = get_review_filename()
    rev_text = read_review_file(rev_file)
    rating, sent = classify_review(rev_text)

    print('Rating: {}'.format(rating), 'Class:', 'Positive' if sent == 1 else 'Negative')
