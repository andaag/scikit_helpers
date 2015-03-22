
# coding: utf-8

# In[2]:

import numpy as np
import sklearn
import sklearn.pipeline
import sklearn.base
import scipy
import pandas as pd


# In[3]:

class FeatureNamedPipeline(sklearn.pipeline.Pipeline):
    """
    Allows generating pipelines with get_feature_names:
    
    sklearn.pipeline.FeatureUnion([
            ("testa", FeatureNamedPipeline([
                ("multiple_columns", PickFeature(select_columns=['Col1', 'Col2'], column_names=data.columns, return_single_row=False)),
                ("csr", ToSparse()),
            ], feature_function="lambda x: x.named_steps["multiple_columns"].select_columns")),

            ("testb", FeatureNamedPipeline([
                ("single_column", PickFeature(select_columns=['TextColumn'], column_names=data.columns, return_single_row=True)),
                ("vect", feature_extraction.text.CountVectorizer(min_df=0.1)),
                ("denseup", ToDense()),
                ("csr", ToSparse()),
            ], feature_function="lambda x: x.named_steps["vect"].vocabulary_"))
    ])
    
    Of course this example doesn't make much sense. You wouldn't want to convert something to dense and then back to sparse.
    
    NB : due to pickle limitations (that are probably fixable) lambdas must for now be strings due to multiprocessing.
    NB2: colunm_names are only needed if the input isn't already a pandas dataframe. Note that you need a very recent 
         pandas and scikit in order to push a dataframe into a pipeline.
    
    """
    def __init__(self, steps, feature_function=None):
        sklearn.pipeline.Pipeline.__init__(self, steps)
        self.feature_function = feature_function
        
    def get_feature_names(self):
        return eval(self.feature_function)(self)


# In[4]:

def make_featurenamed_pipeline(*steps, feature_function=None):
    return FeatureNamedPipeline(sklearn.pipeline._name_estimators(steps), feature_function=feature_function)


# In[5]:

class ConvertToDataframe(sklearn.base.BaseEstimator):
    """
    Converts a numpy array to pandas dataframe in a pipeline/gridsearch.
    Simply put ConvertToDataframe(dataframe.columns) in the top of your pipeline.
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
    Picks features, see FeatureNamedPipeline for full example.
    - select_columns = which columns to select
    - column_names = column names if columns being parsed in are not already a dataframe
    - return_single_row = returns [0] instead of [[0]]
    - return_sparse = return sparse matrix
    """
    def __init__(self, select_columns=[], return_single_row=False, return_sparse=False):
        self.select_columns = select_columns
        self.return_single_row = return_single_row
        self.return_sparse = return_sparse
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
        if self.return_sparse:
            if type(result) is not pd.DataFrame:
                result = pd.DataFrame(result)
                
            if sum(result.dtypes == np.object) > 0:
                print("WARNING: Will not be able to convert this to sparse due to objects in lists!")
                print(result.dtypes[result.dtypes == np.object])
            result = result.to_sparse()
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

