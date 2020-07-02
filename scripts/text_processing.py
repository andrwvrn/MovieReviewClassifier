import pandas as pd
import numpy as np

import re
from functools import lru_cache
from multiprocessing import Pool

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

MORPH = WordNetLemmatizer()

stop_words = set(stopwords.words('english')) 

@lru_cache(maxsize=100000)
def get_normal_form (i):
    return MORPH.lemmatize(i.lower())

def normalize_text(text):
    del_stopwords = [word for word in re.findall(r'[a-zA-Z]{3,}', text) if word not in stop_words]
    normalized = [get_normal_form(word) for word in del_stopwords]
    return ' '.join(normalized)

def process_text(text):
    with Pool(processes=2) as pool:
        text = pool.map(normalize_text, text)
    return text
