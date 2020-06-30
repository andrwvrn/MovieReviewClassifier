import numpy as np
import pandas as pd

from django.conf import settings

import os
import pickle

MOD_FILE = os.path.join(settings.MODELS, 'model.pkl')
VEC_FILE = os.path.join(settings.MODELS, 'vectorizer.pkl')

def get_model():
    with open(MOD_FILE, 'rb') as m, open(VEC_FILE, 'rb') as v:
        model = pickle.load(m)
        vectorizer = pickle.load(v)
    return model, vectorizer

def classify_review(proc_text):
    model, vectorizer = get_model()
    text_feat = vectorizer.transform(proc_text)
    rating = model.predict(text_feat)
    
    return rating[0], int(rating) >= 7

