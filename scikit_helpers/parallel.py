
# coding: utf-8

# In[95]:

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed


# In[96]:

def parallel(df, function, **kwargs):
    chunks = np.array_split(df, 20)
    parsed = Parallel(n_jobs=-1)(delayed(test)(chunks[i], **kwargs) for i in range(len(chunks)))
    
    if len(df.shape) == 1:
        parsed = np.hstack(parsed)
    else:
        parsed = np.vstack(parsed)
    return parsed


# In[97]:

#a = pd.DataFrame(list(range(1000)), columns=["Test"])
#b = a["Test"]
#pd.DataFrame(parallel(a, lambda v:v/2)).head()


# In[ ]:



