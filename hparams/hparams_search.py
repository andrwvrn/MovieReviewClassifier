import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

import pickle
import wandb

SEED = 42
RATIO = 0.3
CV = 3

hyperparameter_defaults = {'ngram_range': 1,
                           'max_df': 0.9,
                           'solver': 'liblinear',
                           'C': 1,
                           'penalty': 'l1',
                          }

wandb.init(project='reviews', config=hyperparameter_defaults)
config = wandb.config

train = pd.read_pickle('../data/train.pkl')

vectorizer = TfidfVectorizer(ngram_range=(1, config['ngram_range']), max_df=config['max_df'])

train_df = train['review_parse']

y_train = train['Rating']
X_train = vectorizer.fit_transform(train_df)

model = OneVsOneClassifier(LogisticRegression(solver=config['solver'], C=config['C'], penalty=config['penalty'], random_state=SEED))

accuracy_ = cross_val_score(model, X_train, y_train, cv=CV, scoring='accuracy').mean()

accuracy_val = {'accuracy_val': accuracy_}
wandb.log(accuracy_val)
