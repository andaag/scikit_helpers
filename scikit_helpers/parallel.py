
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import multiprocessing
import psutil


# In[ ]:

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


# In[ ]:

def parallel(df, func, **kwargs):
    n_jobs = kwargs.pop('n_jobs') if 'n_jobs' in kwargs else 1
    if n_jobs == -1:
        n_jobs = psutil.cpu_count()
    with multiprocessing.Pool(processes=n_jobs) as pool:
        result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, n_jobs)])
    return pd.concat(list(result))


# In[ ]:

#def f(v):
#    return v / 2
#a = pd.DataFrame(list(range(1000)), columns=["Test"])
#a["div"] = parallel(a, f, n_jobs=-1)
#a.head()


# In[ ]:

#dask.dataframe.from_pandas(a, 12).groupby("FakeGroup").apply(dbg).head()


# In[ ]:



