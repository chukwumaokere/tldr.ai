from fastai.imports import *
#from fastai.structured import *
#from structured import *
#from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

import pandas as pd 
from pandas.api.types import is_string_dtype

from sklearn import metrics

import numpy as np 

PATH = "data/bulldozers/"
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, parse_dates=["saledate"])


def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)

def add_datepart(df, fldname):
    fld = df[fldname]
    targ_pre = re.sub('[Dd]ate$', '' , fldname)
    for n in ('Year', 'Month',  'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
                df[targ_pre+n] = getattr(fld.dt,n.lower())
                df[targ_pre+'Elapsed'] = (fld - fld.min()).dt.days
    df_raw.drop(fldname, axis=1, inplace=True)

def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def apply_cats(df, trn):
    for n,c in df.items():
        if trn(n).dtype.name=='category':
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)

df_raw.SalePrice = np.log(df_raw.SalePrice)
add_datepart(df_raw, 'saledate')
train_cats(df_raw)
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
#display_all(df_raw)
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/raw')
#conda install feather-format -c conda-forge

'''
m = RandomForestRegressor(n_jobs=-1)
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
'''
'''


add_datepart(df_raw, 'saledate')
print(df_raw.columns)
m = RandomForestRegressor(n_jobs=1)
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
'''
'''

#df_raw.saleYear.head()
#display_all(df_raw.tail().transpose())

'''
