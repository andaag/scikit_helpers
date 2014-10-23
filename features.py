# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import sklearn
import sklearn.pipeline
import sklearn.base
import scipy
import pandas as pd

# <codecell>

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

# <codecell>

def make_featurenamed_pipeline(*steps, feature_function=None):
    return FeatureNamedPipeline(sklearn.pipeline._name_estimators(steps), feature_function=feature_function)

# <codecell>

class ConvertToDataframe(sklearn.base.BaseEstimator):
    """
    Converts a numpy array to pandas dataframe in a pipeline/gridsearch.
    Simply put ConvertToDataframe(dataframe.columns) in the top of your pipeline.
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, items):
        if type(items) != pd.DataFrame:
            items = pd.DataFrame.from_records(items, columns=self.columns)
        return items

# <codecell>

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
    
    def fit_transform(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then use fit_transform on transformed data using the final
        estimator."""
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        if hasattr(self.steps[-1][-1], 'fit_transform'):
            return self.steps[-1][-1].fit_transform(Xt, y, **fit_params)
        else:
            return self.steps[-1][-1].fit(Xt, y, **fit_params).transform(Xt)
    
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

# <codecell>

class ToDense(sklearn.base.BaseEstimator):
    """
    Converts data to dense, see FeatureNamedPipeline for full example.
    """
    
    def fit(self, items, y=None):
        return self
    
    def transform(self, items):
        return items.todense()

# <codecell>

class ToSparse(sklearn.base.BaseEstimator):
    """
    Converts data to sparse, see FeatureNamedPipeline for full example.
    """
    
    def fit(self, items, y=None):
        return self
    
    def transform(self, items):
        return scipy.sparse.csr_matrix(items)

# <codecell>

class ProbabilityPipeline(sklearn.pipeline.Pipeline):
    """
    Simply a pipeline where the end transform function calls proba/log_proba.
    Can be used as part of a larger pipeline
    """
    def __init__(self, steps, log_proba=False):
        sklearn.pipeline.Pipeline.__init__(self, steps)
        self.log_proba = log_proba
        
    def get_feature_names(self):
        return self.column_names
    
    def fit_transform(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then use fit_transform on transformed data using the final
        estimator."""
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        self.steps[-1][-1].fit(Xt, y, **fit_params)
        return self.transform(Xt)
        
    def transform(self, X):
        if self.log_proba:
            predicted = super(ProbabilityPipeline, self).predict_log_proba(X)
        else:
            predicted = super(ProbabilityPipeline, self).predict_proba(X)
        self.column_names = ["col_"+str(i) for i in range(predicted.shape[1])]
        return predicted

# <codecell>


