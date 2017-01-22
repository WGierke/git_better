import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from classifier import AdaTree, AdaBayes, AdaSVM, GradBoost, DecisionTree, Forest, NaiveBayes, SVM, TheanoNeuralNetwork, TensorFlowNeuralNetwork, XGBoost, TreeBag, SVMBag, get_text_pipeline
from evaluation import complete_columns, get_cleaned_processed_df, eval_classifier, drop_text_features
from nltk.stem.snowball import EnglishStemmer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


JOBLIB_SUFFIX = '.joblib.pkl'
JOBLIB_DESCRIPTION_PIPELINE_NAME = 'best_description_pipeline_4839'
JOBLIB_README_PIPELINE_NAME = 'best_readme_pipeline_5161'
JOBLIB_VOTING_PIPELINE_NAME = 'best_voting_6774_4667'


def find_best_text_pipeline(df_values, labels, pipeline=None, parameters=None):
    if not pipeline:
        pipeline = get_text_pipeline()

    if not parameters:
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
            'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': (0.001, 0.0001, 0.00001, 0.000001),
            'clf__penalty': ('l2', 'elasticnet'),
            'clf__n_iter': (10, 12, 15)
        }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    grid_search.fit(df_values, labels)
    best_parameters = grid_search.best_estimator_.get_params()
    pipeline.set_params(**best_parameters)
    return pipeline


def find_best_repository_classification(df_values, labels, drop_languages=False):
    X_train, X_test, y_train, y_test = train_test_split(df_values, labels, test_size=0.3, random_state=23, stratify=labels)

    # Remove classifiers which you don't want to run and add new ones here
    basic = [DecisionTree, Forest, NaiveBayes, SVM]#], TheanoNeuralNetwork, TensorFlowNeuralNetwork]
    bag = [TreeBag, SVMBag, GradBoost, XGBoost]
    ada = [AdaTree, AdaBayes, AdaSVM]

    val_df = pd.DataFrame.from_csv("data/validation_data.csv")
    val_df = drop_text_features(val_df)
    y_val = val_df.pop("label")
    val_df.fillna(0, inplace=True)

    le = LabelEncoder().fit(np.concatenate((y_train.as_matrix(),y_test.as_matrix(),y_val.as_matrix()), axis=0))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)

    X_train, val_df = complete_columns(X_train, val_df)
    X_test, val_df = complete_columns(X_test, val_df)

    if(drop_languages):
        X_train = drop_languages(X_train)
        X_test = drop_languages(X_test)
        val_df = drop_languages(val_df)

    X_val = val_df.values
    X_train = X_train.values
    X_test = X_test.values

    #results = []
    for classifier in (basic + bag + ada):
        print classifier.__name__

        # Tensorflow needs float32 X data
        if(classifier==TensorFlowNeuralNetwork):
            X_val_buf, X_train_buf, X_test_buf = X_val, X_train, X_test
            X_val, X_train, X_test = val_df.astype(np.float32), X_train.astype(np.float32), X_test.astype(np.float32)

        if(classifier==TheanoNeuralNetwork):
            # Theano needs int32 y data
            y_train_buf, y_test_buf, y_val_buf = y_train, y_test, y_val
            y_train, y_test, y_val = y_train.astype(np.int32), y_test.astype(np.int32), y_val.astype(np.int32)

            # Theano isn't able to cast from int to float64 automatically
            X_val_buf,X_train_buf, X_test_buf = X_val, X_train, X_test
            X_val, X_train, X_test = val_df.astype(np.float64), X_train.astype(np.float64), X_test.astype(np.float64)

        clas = classifier(X_train, y_train)
        clas = clas.fit()

        y_predicted = list(clas.predict(X_test))
        test_score = accuracy_score(y_test, y_predicted)

        y_val_predicted = list(clas.predict(X_val))
        eval_score = accuracy_score(y_val, y_val_predicted)

        print "score on test data: ", test_score
        print "score on evaluation data: ", eval_score
        # could add confusion matrix

        # Theano needs float64 X data
        if(classifier==TensorFlowNeuralNetwork):
            X_val, X_train, X_test = X_val_buf, X_train_buf, X_test_buf

        if(classifier==TheanoNeuralNetwork):
             y_train, y_test, y_val = y_train_buf, y_test_buf, y_val_buf

             X_val, X_train, X_test = X_val_buf, X_train_buf, X_test_buf
    #return results


def drop_languages(df):
    for c in df.columns:
        if "LANGUAGE" in c:
            df = df.drop(c, axis=1)
    return df

def save_pickle(model, filename):
    joblib.dump(model, filename + JOBLIB_SUFFIX, compress=9)


def load_pickle(filename):
    return joblib.load(filename + JOBLIB_SUFFIX)


def drop_defect_rows(df):
    return df.loc[df["mentionableUsers"] > 0]


def get_undersample_df(df):
    if df is None:
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
    assert len(samples_df.groupby('label').count().iloc[:, 0].unique()), 1
    return samples_df.reset_index()
