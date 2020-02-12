from fastai.imports import *
#from fastai.structured import *
#from structured import *
#from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
import os

import pandas as pd 
from pandas.api.types import *

from sklearn import metrics

import numpy as np 


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

def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
            if not ignore_flds: ignore_flds=[]
            if not skip_flds: skip_flds=[]
            if subset: df = get_sample(df,subset)
            else: df = df.copy()
            ignored_flds = df.loc[:, ignore_flds]
            df.drop(ignore_flds, axis=1, inplace=True)
            if preproc_fn: preproc_fn(df)
            if y_fld is None: y = None
            else:
                if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
                y = df[y_fld].values
                skip_flds += [y_fld]
            df.drop(skip_flds, axis=1, inplace=True)

            if na_dict is None: na_dict = {}
            else: na_dict = na_dict.copy()
            na_dict_initial = na_dict.copy()
            for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
            if len(na_dict_initial.keys()) > 0:
                df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
            if do_scale: mapper = scale_vars(df, mapper)
            for n,c in df.items(): numericalize(df, c, n, max_n_cat)
            df = pd.get_dummies(df, dummy_na=True)
            df = pd.concat([ignored_flds, df], axis=1)
            res = [df, y, na_dict]
            if do_scale: res = res + [mapper]
            return res

def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1

def scale_vars(df, mapper):
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

def get_nn_mappers(df, cat_vars, contin_vars):
    # Replace nulls with 0 for continuous, "" for categorical.
    for v in contin_vars: df[v] = df[v].fillna(df[v].max()+100,)
    for v in cat_vars: df[v].fillna('#NA#', inplace=True)

    # list of tuples, containing variable and instance of a transformer for that variable
    # for categoricals, use LabelEncoder to map to integers. For continuous, standardize
    cat_maps = [(o, LabelEncoder()) for o in cat_vars]
    contin_maps = [([o], StandardScaler()) for o in contin_vars]

def is_date(x): return np.issubdtype(x.dtype, np.datetime64)

def get_sample(df,n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

def combine_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))

def set_plot_sizes(sml, med, big):
    plt.rc('font', size=sml)          # controls default text sizes
    plt.rc('axes', titlesize=sml)     # fontsize of the axes title
    plt.rc('axes', labelsize=med)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('legend', fontsize=sml)    # legend fontsize
    plt.rc('figure', titlesize=big)  # fontsize of the figure title

def parallel_trees(m, fn, n_jobs=8):
    return list(ProcessPoolExecutor(n_jobs).map(fn, m.estimators_))

def split_vals(a,n): return a[:n].copy(), a[n:].copy()

def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(n):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid), m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

if (os.path.isfile('tmp/raw')):
    print("reading from raw output")
    df_raw = pd.read_feather('tmp/raw')
else:
    print("reading from csv")
    df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, parse_dates=['saledate'])
    df_raw.SalePrice = np.log(df_raw.SalePrice)
    add_datepart(df_raw, 'saledate')
    train_cats(df_raw)
    df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)

#display_all(df_raw)
#display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
#os.makedirs('tmp', exist_ok=True)
#df_raw.to_feather('tmp/raw')
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
df, y, nas = proc_df(df_raw, 'SalePrice')

n_valid = 12000
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df, y)

#%time m.fit(X_train, y_train)
print_score(m)