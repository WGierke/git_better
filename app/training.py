from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def get_best_text_pipeline(df_values, labels, pipeline=None, params=None):
    if not pipeline:
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier()),
        ])

    if not params:
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': (0.00001, 0.000001),
            'clf__penalty': ('l2', 'elasticnet'),
            'clf__n_iter': (10, 50, 80),
        }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0)
    grid_search.fit(df_values, labels)
    best_parameters = grid_search.best_estimator_.get_params()
    pipeline.set_params(**best_parameters)
    return pipeline
