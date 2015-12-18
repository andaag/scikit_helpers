# -*- coding: utf-8 -*-

from datetime import datetime, timedelta 
import os
import pandas as pd
from sklearn.externals import joblib
import random
import time

import hashlib

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
            
            isdf = os.path.exists(filename + ".df")
            isjoblib = os.path.exists(filename)
            
            if isjoblib or isdf:
                if isdf:
                    #print("Loading pandas data...")
                    res = pd.read_pickle(filename + ".df")
                else:
                    #print("Loading joblib data...")
                    res = joblib.load(filename)
                print("Cache hit", (time.time() - s), "hashtime", hashtime)
                return res
            else:
                res = func(*args, **kwargs)
                if type(res) == pd.DataFrame or type(res) == pd.Series:
                    #print("Writing pandas data...")
                    res.to_pickle(filename + ".df")
                else:
                    #print("Writing generic data...")
                    joblib.dump(res, filename, compress=3)
                print("Cache miss", (time.time() - s), "hashtime", hashtime)
                return res
        return inner


# In[ ]:

#@cached(filename="tmp/cache/testcache")
#def test(a, test="asd"):
#    return "The argument is", a, random.randint(1,100)


# In[ ]:

#test("simple")
#test("A", test="bsd")
#test(True)
#test("B")
#test({"a":5})


# In[ ]:



