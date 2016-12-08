import numpy as np
import pandas as pd
from sklearn.externals import joblib
from app.evaluation import get_cleaned_processed_df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import EnglishStemmer

JOBLIB_SUFFIX = '.joblib.pkl'
JOBLIB_DESCRIPTION_PIPELINE_NAME = 'best_description_pipeline'
JOBLIB_README_PIPELINE_NAME = 'best_readme_pipeline_4516'


def stemmed_words(doc):
    stemmer = EnglishStemmer()
    analyzer = CountVectorizer().build_analyzer()
    return (stemmer.stem(w) for w in analyzer(doc))


def get_best_text_pipeline(df_values, labels, pipeline=None, params=None):
    if not pipeline:
        pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words='english', analyzer=stemmed_words)),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss="log")),
        ])

    if not params:
        parameters = {
            #'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
            #'tfidf__use_idf': (True, False),
            #'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': (0.001, 0.0001, 0.00001, 0.000001),
            #'clf__penalty': ('l2', 'elasticnet'),
            'clf__n_iter': (8, 10, 12),
        }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    grid_search.fit(df_values, labels)
    best_parameters = grid_search.best_estimator_.get_params()
    pipeline.set_params(**best_parameters)
    return pipeline


def save_pickle(model, filename):
    joblib.dump(model, filename + JOBLIB_SUFFIX, compress=9)


def load_pickle(filename):
    return joblib.load(filename + JOBLIB_SUFFIX)


def get_undersample_df(df):
    df = get_cleaned_processed_df()
    samples_df = pd.DataFrame(columns=df.columns)
    label_counts = df.groupby('label').count().iloc[:, 0]
    minimum_count = label_counts.min()

    for label in label_counts.index:
        indices = df[df.label == label].index
        sampled_indices = np.random.choice(indices, minimum_count, replace=False)
        samples = df.loc[sampled_indices]
        samples_df = samples_df.append(samples)

    assert len(samples_df), minimum_count * label_counts
    return samples_df
