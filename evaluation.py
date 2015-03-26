
# coding: utf-8

# In[ ]:

from sklearn.cross_validation import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import pylab as pl
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit

def _learning_curve_evaluate(clf, X, Y, train, test, score):
    clf.fit(X[train], Y[train])
    if score:
        return clf.score(X[train], Y[train]), clf.score(X[test], Y[test])
    return (clf.predict(X[train]), Y[train]), (clf.predict(X[test]), Y[test])
    

def learning_curve(clf, X, Y, train_sizes=np.logspace(2,3,5,6).astype(np.int), n_iter=6, n_jobs=4, stratified=False, scorer=None):
    """
    Generates a learning curve.
    
    >>> learning_curve(RandomForestClassifier(n_jobs=-1), X, Y)
    
    train_sizes is an array consisting of the amount of data to use for each iteration.
    n_iter is number of iterations to do in each shuffle split.
    
    This class is roughly 99.3% stolen from Oliver Grisel (http://ogrisel.com/)
    """
    if type(Y) is not np.array:
        Y = np.array(Y)
    if type(train_sizes) is not np.array:
        train_sizes = np.array(train_sizes)
    
    if max(train_sizes) > len(Y):
        raise Exception("Train sizes > len(Y)")
    
    train_scores = np.zeros((train_sizes.shape[0], n_iter), dtype=np.float)
    test_scores = np.zeros((train_sizes.shape[0], n_iter), dtype=np.float)
    for i, train_size in enumerate(train_sizes):
        cv = None
        if stratified:
            cv = StratifiedShuffleSplit(Y, n_iter=n_iter, train_size=train_size)
        else:
            cv = ShuffleSplit(len(Y), n_iter=n_iter, train_size=train_size)
        
        result = Parallel(n_jobs=n_jobs)(delayed(_learning_curve_evaluate)(clone(clf), X, Y, train, test, scorer is None) for (train, test) in cv)
        for j, (train, test) in enumerate(result):
            if scorer is not None:
                train = scorer(train[1], train[0])
                test = scorer(test[1], test[0])
            train_scores[i, j] = train
            test_scores[i, j] = test

            
    def plot_learning_curve():
        from scipy.stats import sem
        mean_train = np.mean(train_scores, axis=1)
        confidence = sem(train_scores, axis=1) * 2
        plt.fill_between(train_sizes, mean_train - confidence, mean_train + confidence, color='b', alpha=.2)
        plt.plot(train_sizes, mean_train, 'o-k', c='b', label='Train score')

        mean_test = np.mean(test_scores, axis=1)
        confidence = sem(test_scores, axis=1) * 2
        plt.fill_between(train_sizes, mean_test - confidence, mean_test + confidence, color='g', alpha=.2)
        plt.plot(train_sizes, mean_test, 'o-k', c='g', label='Test score')

        plt.xlabel('Training set size')
        plt.ylabel('Score')
        plt.xlim(0, max(train_sizes))
        
        ylimiter = max(np.array(train_scores).max(), np.array(test_scores).max(), 1.01)
        
        plt.ylim((None, ylimiter))
        plt.legend(loc='best')
        plt.title('Main train and test scores')
    
    plot_learning_curve()
    return train_scores, test_scores

def boxplot_parameters(clf):
        """Plot boxplot of RandomizedSearchCV parameters. Idea and some code stolen from:
        https://github.com/ogrisel/parallel_ml_tutorial
        Utilities for Parallel Model Selection with IPython
        Author: Olivier Grisel <olivier@ogrisel.com>
        Licensed: Simplified BSD
        
        I generally start off with a large parameter grid (much larger than example), and fairly low n_iter.
        Then use the results of the boxplot to filter out some of the parameters that I don't need, before I
        run a proper gridsearch over the parameters that look good.
        
        >>> clf = RandomizedSearchCV(SVC(), {"C":[0.1, 1,10, 100], "kernel":["rbf"], "gamma":[0.0, 0.1]})
        >>> clf.fit(X_train, y_train)
        >>> boxplot_parameters(clf)
        """
        results = clf.grid_scores_

        n_rows = len(clf.param_distributions)
        pl.figure()
        for i, (param_name, param_values) in enumerate(clf.param_distributions.items()):
            pl.subplot(n_rows, 1, i + 1)
            scores_per_value = []
            for param_value in param_values:
                scores = [r.cv_validation_scores for r in results if r.parameters[param_name] == param_value]
                scores_per_value.append(scores)
            
            widths = 0.25
            positions = np.arange(len(param_values)) + 1
            offset = 0        
            
            offset = 0.175
            pl.boxplot(scores_per_value, widths=widths, positions=positions - offset)

            pl.xticks(np.arange(len(param_values)) + 1, param_values)
            pl.xlabel(param_name)
            pl.ylabel("Val. Score")
            plt.show()


# In[ ]:



