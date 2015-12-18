# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve

from scipy.stats import sem
import pylab as pl
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit

"""
Shamelessly stolen from http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html. I just keep it here for quicker access.
"""

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring=None):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

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