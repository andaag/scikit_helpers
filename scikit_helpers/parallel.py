
# coding: utf-8

# In[99]:

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
import psutil


# In[102]:

def parallel(df, function, n_splits=psutil.cpu_count() * 4, n_jobs=-1, **kwargs):
    chunks = np.array_split(df, n_splits)
    parsed = Parallel(n_jobs=n_jobs)(delayed(test)(chunks[i], **kwargs) for i in range(len(chunks)))
    
    if len(df.shape) == 1:
        parsed = np.hstack(parsed)
    else:
        parsed = np.vstack(parsed)
    return parsed


# In[104]:

#a = pd.DataFrame(list(range(1000)), columns=["Test"])
#b = a["Test"]
#pd.DataFrame(parallel(a, lambda v:v/2)).head()


# In[ ]:



