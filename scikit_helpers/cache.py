
# coding: utf-8

# In[1]:

from datetime import datetime, timedelta 
import os
import pandas as pd
from sklearn.externals import joblib
import random
import time

import hashlib


# In[2]:

class cached(object):
    def __init__(self, *args, **kwargs):
        self.filename = kwargs.get("filename")
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        assert len(self.filename) > 2

    def __call__(self, func):
        def inner(*args, **kwargs):
            m = hashlib.md5()
            
            def quickhashdict(values):
                items = list(values.items())
                #if len(items) < 100:
                #    return frozenset(sorted(v.items())
                s = ""
                size = len(items)
                s += str(size)
                if size > 0:
                    s += str(items[0])
                    s += str(items[int(size / 2)])
                    s += str(items[size - 1])
                return s
            
            def performhash(v):
                if type(v) == dict:
                    return m.update(quickhashdict(v).encode("utf-8"))
                else:
                    return m.update(str(v).encode("utf-8"))
            s = time.time()
            
            for v in args:
                performhash(v)
                
            for k,v in sorted(kwargs.items()):
                m.update(str(k).encode("utf-8"))
                performhash(v)
                
            filename = self.filename + "." + m.hexdigest()
            hashtime = time.time() - s
            print("Filename", filename)
            if not os.path.exists(filename):
                res = func(*args, **kwargs)
                joblib.dump(res, filename, compress=3)
                print("Cache miss", (time.time() - s), "hashtime", hashtime)
                return res
            else:
                res = joblib.load(filename)
                print("Cache hit", (time.time() - s), "hashtime", hashtime)
                return res
        return inner


# In[3]:

#@cached(filename="tmp/cache/testcache")
#def test(a, test="asd"):
#    return "The argument is", a, random.randint(1,100)


# In[4]:

#test("simple")
#test("A", test="bsd")
#test(True)
#test("B")
#test({"a":5})


# In[ ]:



