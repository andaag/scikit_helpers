
# coding: utf-8

# In[2]:

import numpy as np
import sklearn
import sklearn.pipeline
import sklearn.base
import scipy
import pandas as pd


# In[3]:

class FeatureNamePipeline(sklearn.pipeline.Pipeline):
    """
    Very simple class that wraps pipeline and calls get_feature_names on the last level of the pipeline. 
    If you have a pipeline ending in tfidfvectorizer for example this will just work and return feature names as expected
    """
    def __init__(self, steps, feature_function=None):
        sklearn.pipeline.Pipeline.__init__(self, steps)
        
    def get_feature_names(self):
        """Get feature names from last step in pipeline
        Returns
        -------
        feature_names : list of strings
            Names of the features produced by last step in transform.
        """
        last_step = self.steps[-1]
        transformer = last_step[1]
        name = last_step[0]
        if not hasattr(transformer, 'get_feature_names'):
            raise AttributeError("Transformer %s does not provide"
                                 " get_feature_names." % str(name))
        return transformer.get_feature_names()


# In[4]:

def make_featurename_pipeline(*steps):
    return FeatureNamePipeline(sklearn.pipeline._name_estimators(steps))


# In[5]:

class ConvertToDataframe(sklearn.base.BaseEstimator):
    """
    Converts a numpy array to pandas dataframe in a pipeline/gridsearch.
    Simply put ConvertToDataframe(dataframe.columns) in the top of your pipeline.
    
     - should not be needed on recent versions of sklearn.
    """
    def __init__(self, columns):
        if (type(columns) == pd.core.index.Index):
            columns = columns.values
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, items):
        if type(items) != pd.DataFrame:
            items = pd.DataFrame.from_records(items, columns=self.columns)
        return items


# In[6]:

class PickFeature(sklearn.base.BaseEstimator):
    """
    Picks features.
    - select_columns = which columns to select
    - column_names = column names if columns being parsed in are not already a dataframe
    - return_single_row = returns [0] instead of [[0]]
    """
    def __init__(self, select_columns=[], return_single_row=False):
        if type(select_columns) == str:
            select_columns = [select_columns]
        self.select_columns = select_columns
        self.return_single_row = return_single_row
        assert len(self.select_columns) > 0
        assert not self.return_single_row or len(self.select_columns) == 1
    
    def get_feature_names(self):
        return self.select_columns
    
    def fit(self, items, y=None):
        return self
    
    def transform(self, items):
        if type(items) is not pd.DataFrame:
            raise Exception("PickFeature requires a dataframe, put ConvertToDataframe(column_names) in the start of the pipeline")
        if self.return_single_row:
            result = items[self.select_columns[0]]
        else:
            result = items[self.select_columns]
        return result


# In[7]:

class ToDense(sklearn.base.BaseEstimator):
    """
    Converts data to dense, see FeatureNamedPipeline for full example.
    """
    
    def fit(self, items, y=None):
        return self
    
    def transform(self, items):
        return items.todense()


# In[8]:

class ToSparse(sklearn.base.BaseEstimator):
    """
    Converts data to sparse, see FeatureNamedPipeline for full example.
    """
    
    def fit(self, items, y=None):
        return self
    
    def transform(self, items):
        return scipy.sparse.csr_matrix(items)


# In[9]:

class ProbabilityEstimator(sklearn.base.BaseEstimator):
    """
    ProbabilityEstimator(sklearn.linear_model.SGDClassifier(loss='modified_huber'))
    """
    def __init__(self, base_estimator=None, log_proba=False):
        self.base_estimator = base_estimator
        self.log_proba = log_proba
        
    def fit(self, X, y=None):
        self.base_estimator.fit(X, y)
        return self
        
    def transform(self, X):        
        if self.log_proba:
            return self.base_estimator.predict_log_proba(X)
        else:
            return self.base_estimator.predict_proba(X)


# In[ ]:



