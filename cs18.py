from fastai.imports import *
#from fastai.structured import *
#from structured import *
#from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics

import numpy as np 

PATH = "data/bulldozers/"
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, parse_dates=["saledate"])


def display_all(df):
    display(df)

def add_datepart(df, fldname):
    fld = df[fldname]
    targ_pre = re.sub('[Dd]ate$', '' , fldname)
    for n in ('Year', 'Month',  'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
                df[targ_pre+n] = getattr(fld.dt,n.lower())
                df[targ_pre+'Elapse'] = (fld - fld.min()).dt.days
                df.drop(fldname, axis=1, inplace=True)

def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()


#df_raw.saleYear.head()
display_all(df_raw.tail().transpose())

m = RandomForestRegressor(n_jobs=1)
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)

add_datepart(df_raw, 'saledate')