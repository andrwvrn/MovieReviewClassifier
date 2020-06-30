import pandas as pd
import numpy as np

import re
from functools import lru_cache
from multiprocessing import Pool

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

MORPH = SnowballStemmer('english')

stop_words = set(stopwords.words('english')) 

@lru_cache(maxsize=100000)
def get_normal_form (i):
    return MORPH.stem(i.lower())

def normalize_text(text):
    normalized = [get_normal_form(word) for word in re.findall(r'[a-zA-Z]{3,}', text)]
    return ' '.join([word for word in normalized if word not in stop_words])

def process_text(text):
    with Pool(processes=2) as pool:
        text = pool.map(normalize_text, text)
    return text
