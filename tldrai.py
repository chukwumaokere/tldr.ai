import fastai
from fastai import *
from fastai.text import *
import pandas as pd
import numpy as np 
from functools import partial 
import io 
import os 

from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

print(documents)
df = pd.DataFrame({'label':dataset.target, 'text':dataset.data})
print(df.shape) 