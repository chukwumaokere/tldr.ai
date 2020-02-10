
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics

import numpy as np 

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
