
# coding: utf-8

# In[177]:

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
import psutil


# In[178]:

def parallel(df, function, n_splits=psutil.cpu_count() * 4, n_jobs=-1, **kwargs):
    ## nb : currently only supports returning single column, can work around that with reshaping though
    chunks = np.array_split(df, n_splits)
    parsed = Parallel(n_jobs=n_jobs)(delayed(function)(chunks[i], **kwargs) for i in range(len(chunks)))
    parsed = np.vstack(parsed)
    return parsed


# In[181]:

#def f(v):
#    return v / 2

#a = pd.DataFrame(list(range(1000)), columns=["Test"])
#a["div"] = parallel(a, f, n_jobs=-1)
#a.head()


# In[ ]:




# In[ ]:



