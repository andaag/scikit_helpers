
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import multiprocessing
import psutil


# In[2]:

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


# In[3]:

def _apply_grouped_df(args):
    grp, df, func, kwargs = args
    r = df.apply(func, **kwargs)
    if type(r) is pd.Series:
        r.name = grp
    return r


# In[4]:

def parallel(df, func, **kwargs):
    n_jobs = kwargs.pop('n_jobs') if 'n_jobs' in kwargs else 1
    if n_jobs == -1:
        n_jobs = psutil.cpu_count()
    with multiprocessing.Pool(processes=n_jobs) as pool:
        if type(df) is pd.core.groupby.DataFrameGroupBy:
            result = pool.map(_apply_grouped_df, [(grpname, d, func, kwargs)
                for grpname, d in df])
            if len(result) == 0:
                return result
            if type(result[0]) is pd.Series:
                return pd.concat(result, axis=1).T
            return pd.concat(result)
        else:
            result = pool.map(_apply_df, [(d, func, kwargs)
                for d in np.array_split(df, n_jobs)])
            return pd.concat(list(result))


# In[5]:

#def f(v):
#    return v / 2
#a = pd.DataFrame(list(range(1000)), columns=["Test"])
#a["div"] = parallel(a, f, n_jobs=-1)
#a.head()


# In[ ]:

#def grpfunc(v):
#    return v.sum()


# In[6]:

#a["FakeGroup"] = a["Test"].apply(lambda v:int(v % 2 == 0))
#a.groupby("FakeGroup").apply(grpfunc)


# In[8]:

#parallel(a.groupby("FakeGroup"), grpfunc, n_jobs=-1)

