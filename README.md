scikit_helpers
==============

My personal tools for working with scikit/ML. For now it's only my feature building tools with scikit pipelines and pandas dataframes.

	pipe = sklearn.pipeline.Pipeline([
	    ("features", sklearn.pipeline.FeatureUnion([
	                ("single_features", PickFeature(select_columns=["ColumnA","ColumnB"], return_sparse=True)),
	                ("text_pipeline", FeatureNamedPipeline([
	                    ("pick", PickFeature(select_columns=['Annonseoverskrift'], return_single_row=True)),
	                    ("vect", sklearn.feature_extraction.text.CountVectorizer()),
	                    ("kbest", sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2)),
	                	], feature_function='lambda x: np.asarray(x.named_steps["vect"].get_feature_names())[x.named_steps["kbest"].get_support()]'))
	             ])),
	    ("svc", sklearn.svm.LinearSVC())
	    ])
	pipe.fit(X, y) #X in this case is a pandas dataframe.
	pipe.predict(X)


Now you can easily run a gridsearch tuning parameters for everything from the svc in the bottom, kbest/the vectorizer etc. It also supports pipe.named_steps["features"].get_feature_names(). 

In this example I convert the entire featureunion to sparse. Of course you can skip the return_sparse function, but then you would have to stick a ToDense() feature after SelectKBest in order for the FeatureUnion to be able to merge them.