# code partly from our previous project Data Mining Cup, https://github.com/AlexImmer/run-dmc
import logging
import numpy as np, random
import pandas as pd
from nltk.stem.snowball import EnglishStemmer
from operator import itemgetter
from preprocess import ColumnSumFilter, ColumnStdFilter, PolynomialTransformer
from scipy.sparse import csr_matrix
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

TOP_NUMERIC_FEATURES = 30

try:
    import theanets as tn
except Exception, e:
    logging.error("Can't import Theano: " + str(e))

try:
    import tensorflow as tf
    import tensorflow.contrib.learn as sklearn
except ImportError:
    print('Tensorflow not installed')


class GIClassifier(object):
    clf = None

    def __call__(self, X):
        if self.tune_parameters:
            print(self.clf.get_params().keys())
            try:
                self.estimate_parameters_with_random_search()
            except Exception as e:
                print(e)
                pass
        self.fit()
        return self.predict(X)

    def report(self, grid_scores, n_top=3):
        top_scores = sorted(
            grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                score.mean_validation_score, np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

    def estimate_parameters_with_random_search(self):
        random_search = RandomizedSearchCV(self.clf, param_distributions=self.param_dist_random,
                                           n_iter=30)
        random_search.fit(self.X, self.Y)
        print("Random Search")
        self.report(random_search.grid_scores_)

    def fit(self, df, Y, tune_parameters=False):
        self.tune_parameters = tune_parameters
        self.clf.fit(df[df.select_dtypes(include=[np.number]).columns], Y)
        return self

    def predict(self, df):
        return self.clf.predict(df[df.select_dtypes(include=[np.number]).columns])

    def predict_proba(self, df):
        return self.clf.predict_proba(df[df.select_dtypes(include=[np.number]).columns])

    def set_params(self, **args):
        return self.clf.set_params(**args)

    def get_params(self, **args):
        return self.clf.get_params(**args)

    def score(self, df, Y):
        return self.clf.score(df[df.select_dtypes(include=[np.number]).columns], Y)


class DecisionTree(GIClassifier):
    def __init__(self, **args):
        self.param_dist_random = {'max_depth': sp_randint(1, 100),
                                      'min_samples_leaf': sp_randint(1, 150),
                                      'criterion': ['entropy', 'gini']}
        self.clf = DecisionTreeClassifier(**args)

class Forest(GIClassifier):
    def __init__(self, **args):
        self.param_dist_random = {'max_depth': sp_randint(1, 100),
                                      'min_samples_leaf': sp_randint(1, 100),
                                      'criterion': ['entropy', 'gini']}
        self.clf = RandomForestClassifier(**args)


class KNeighbors(GIClassifier):
    def __init__(self, **args):
        self.param_dist_random = {'leaf_size':sp_randint(20, 50),
                                    'n_neighbors': sp_randint(4, 30)}
        self.clf = KNeighborsClassifier(**args)


class NaiveBayes(GIClassifier):
    def __init__(self, **args):
        self.clf = GaussianNB(**args)


class SVM(GIClassifier):
    def __init__(self, **args):
        self.param_dist_random = {'shrinking': [True, False],
                                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                      'degree': sp_randint(2, 5)}
        self.clf = SVC(**args)


class LinearSVM(GIClassifier):
    def __init__(self, **args):
        self.param_dist_random = {'shrinking': [True, False],
                                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                      'degree': sp_randint(2, 5)}
        self.clf = LinearSVC(**args)


class BagEnsemble(GIClassifier):
    base_estimator = None
    estimators = 20
    max_features = .5
    max_samples = .5

    def __init__(self, **args):
        self.param_dist_random = {'max_features': sp_randint(1, self.X.shape[1]),
                                      'n_estimators': sp_randint(1, 100)}
        self.clf = BaggingClassifier(base_estimator=self.base_estimator, n_estimators=self.estimators, n_jobs=-1,
                                     max_samples=self.max_samples, max_features=self.max_features)


class TreeBag(BagEnsemble):
    def __init__(self, **args):
        self.base_estimator = DecisionTreeClassifier(**args)


class SVMBag(GIClassifier):

    def __init__(self, **args):
        self.classifier = SVC(decision_function_shape='ovo')
        self.clf = BaggingClassifier(**args)

class AdaBoostEnsemble(GIClassifier):

    def __init__(self, **args):
        self.param_dist_random = {'n_estimators': sp_randint(1, 1000),
                                      'algorithm': ['SAMME', 'SAMME.R'],
                                      'learning_rate': 100* random.random()}
        self.param_dist_grid = {'n_estimators': [100, 200, 400, 900, 1000],
                                    'algorithm': ['SAMME', 'SAMME.R'],
                                    'learning_rate': [.1, .2, 0.25, .3,
                                                      .4, .5, .6]}
        self.clf = AdaBoostClassifier(**args)


class AdaTree(AdaBoostEnsemble):

    def __init__(self, **args):
        self.clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())


class AdaBayes(AdaBoostEnsemble):

    def __init__(self, **args):
        self.clf = AdaBoostClassifier(base_estimator=BernoulliNB)


class AdaSVM(AdaBoostEnsemble):

    def __init__(self, **args):
        self.clf = AdaBoostClassifier(base_estimator=SVC)


class GradBoost(GIClassifier):

    def __init__(self, **args):
        self.clf = GradientBoostingClassifier(**args)


class XGBoost(GIClassifier):
    def __init__(self, **args):
        self.param_dist_random = {'max_depth': sp_randint(1, 20),
                                      'n_estimators' : sp_randint(50, 200)}
        self.clf = XGBClassifier(**args)

    def score(self, df, Y):
        return self.clf.score(df[df.select_dtypes(include=[np.number]).columns].values, Y)

    def predict(self, df):
        return self.clf.score(df[df.select_dtypes(include=[np.number]).columns].values)


class TheanoNeuralNetwork(GIClassifier):
   def __init__(self, X, Y, tune_parameters=False):
       super(TheanoNeuralNetwork, self).__init__(X, Y, tune_parameters=False)
       input_layer, output_layer = self.X.shape[1], len(np.unique(Y))
       inp = tn.layers.base.Input(size=input_layer, sparse='csr')
       self.clf = tn.Classifier(layers=[inp,
                                        (100, 'linear'), (50, 'norm:mean+relu'),
                                        output_layer])

   def fit(self):
       self.clf.train((self.X, self.Y), algo='sgd', learning_rate=.05, momentum=0.9)
       return self


class TensorFlowNeuralNetwork(GIClassifier):
    steps = 20000
    # Deprecated with new DNNClassifier-API
    # learning_rate = 0.05
    hidden_units = [100, 100]
    optimizer = 'SGD'

    def __init__(self, X, Y, tune_parameters=False):
        super(TensorFlowNeuralNetwork, self).__init__(X, Y, tune_parameters=False)
        self.X = X #.todense()  # TensorFlow/Skflow doesn't support sparse matrices
        # convert string labels into numerical labels
        self.Y = pd.factorize(Y)[0]
        output_layer = len(np.unique(Y))
        if tune_parameters:
            self.param_dist_random = {'learning_rate': random.random(100),
                                      'optimizer': ['Adam'],
                                      'hidden_units': [sp_randint(50, 500), sp_randint(50, 500)]}

        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=self.X.shape[1])]
        self.clf = sklearn.DNNClassifier(hidden_units=self.hidden_units, feature_columns=feature_columns,
                                          n_classes=output_layer, optimizer='Adam', model_dir="log/dnn/")
                                                  #optimizer=self.optimizer) model_dir="",
    def fit(self):
       self.clf.fit(x=self.X, y=self.Y, steps=self.steps)
       return self

    def predict(self, X):
        #X = X.todense()  # TensorFlow/Skflow doesn't support sparse matrices
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
        #.todense())

class MetaClassifier(GIClassifier):
    important_column = None
    fill_character = None

    def fit(self, df_origin, Y, tune_parameters=False):
        df = df_origin.copy()
        self.tune_parameters = tune_parameters
        df[self.important_column].fillna(self.fill_character, inplace=True)
        self.clf.fit(df[self.important_column].values, Y)
        return self

    def predict(self, df_origin):
        df = df_origin.copy()
        df[self.important_column].fillna(self.fill_character, inplace=True)
        return self.clf.predict(df[self.important_column].values)

    def predict_proba(self, df_origin):
        df = df_origin.copy()
        df[self.important_column].fillna(self.fill_character, inplace=True)
        return self.clf.predict_proba(df[self.important_column].values)

    def score(self, df_origin, Y):
        df = df_origin.copy()
        df[self.important_column].fillna(self.fill_character, inplace=True)
        return self.clf.score(df[self.important_column], Y)


class DescriptionClassifier(MetaClassifier):
    important_column = "description"
    fill_character = ''

    def __init__(self, **args):
        self.clf = get_text_pipeline(**args)


class ReadmeClassifier(MetaClassifier):
    important_column = "readme"
    fill_character = ''

    def __init__(self, **args):
        self.clf = get_text_pipeline(**args)


class NumericEnsembleClassifier(MetaClassifier):
    fill_character = 0

    def __init__(self, **args):
        self.clf = get_voting_classifier(**args)
        self.useful_features = []

    def fit(self, df_origin, Y, tune_parameters=False):
        df = df_origin.copy()
        self.tune_parameters = tune_parameters
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = df[self.numeric_columns]
        model = ExtraTreesClassifier(n_jobs=-1)
        df.fillna(self.fill_character, inplace=True)
        model.fit(df, Y)
        zipped = zip(df.columns, model.feature_importances_)
        zipped.sort(key=lambda x: x[1], reverse=True)
        self.useful_features = [x[0] for x in zipped[:TOP_NUMERIC_FEATURES]]
        self.useful_features = list(set(self.useful_features))
        df = keep_useful_features(df, self.useful_features)
        self.clf.fit(df, Y)
        return self

    def predict(self, df_origin):
        df = df_origin.copy()
        df = self.transform_to_fitted_features(df)
        return self.clf.predict(df)

    def predict_proba(self, df_origin):
        df = df_origin.copy()
        df = self.transform_to_fitted_features(df)
        return self.clf.predict_proba(df)

    def score(self, df_origin, Y):
        df = df_origin.copy()
        df = self.transform_to_fitted_features(df)
        return self.clf.score(df, Y)

    def transform_to_fitted_features(self, df_origin):
        df = df_origin.copy()
        df = df.fillna(self.fill_character)
        df = keep_useful_features(df, self.useful_features)
        return df


class EnsembleAllNumeric(MetaClassifier):
    """Fits a RandomForestClassifier on the features where the text features have been transformed to numeric ones"""
    fill_character = 0

    def __init__(self, **args):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(**args)

    def fit(self, df_origin, Y, tune_parameters=False):
        df = df_origin.copy()
        df = normalize(df)
        df = self.transform(df)
        self.useful_features = list(df.columns)
        self.clf.fit(df, Y)
        return self

    def predict(self, df_origin):
        df = df_origin.copy()
        df = self.transform(df)
        df = keep_useful_features(df, self.useful_features)
        return self.clf.predict(df)

    def predict_proba(self, df_origin):
        df = df_origin.copy()
        df = self.transform(df)
        df = keep_useful_features(df, self.useful_features)
        return self.clf.predict_proba(df)

    def score(self, df_origin, Y):
        df = df_origin.copy()
        df = self.transform(df)
        df = keep_useful_features(df, self.useful_features)
        return self.clf.score(df, Y)

    def transform(self, df):
        text_columns = df.select_dtypes(exclude=[np.number]).columns
        cv = CountVectorizer(token_pattern="[a-zA-Z0-9.:]{3,}", min_df=0.001)
        for c in text_columns:
            df[c] = df[c].astype(str)
            matrix = cv.fit_transform(df[c]).todense()
            features = cv.get_feature_names()
            normalized_matrix = matrix / matrix.sum(axis=1, dtype=float)
            for i in range(len(features)):
                df[c + "_" + features[i]] = normalized_matrix[:, i]
            df.drop(c, axis=1, inplace=True)
            del matrix
            del normalized_matrix
        del cv
        return df.fillna(0)


def get_text_pipeline(**args):
    ppl = Pipeline([
        ('vect', CountVectorizer(stop_words='english', analyzer=stemmed_words, token_pattern='[a-zA-Z]{3,}')),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss="log", n_jobs=-1)),
    ])

    if args:
        ppl.set_params(**args)
    return ppl


def stemmed_words(doc):
    stemmer = EnglishStemmer()
    analyzer = CountVectorizer().build_analyzer()
    return (stemmer.stem(w) for w in analyzer(doc))


def get_voting_classifier(**args):
    voting_clf = VotingClassifier(voting='soft', estimators=[
        ('clf_bayes', NaiveBayes()),
        ('clf_tree', DecisionTree()),
        ('clf_forest', Forest(n_jobs=-1)),
        ('clf_kneighbors', KNeighbors()),
        ('clf_svm', SVM(kernel='rbf', probability=True)),
        #('clf_linear_svm', LinearSVM()),
        ('clf_grad_boost', GradBoost())])
        #('clf_xgboost', XGBoost())])
        # ('clf_bag_ensemble', BagEnsemble()),
        #('clf_treebag', TreeBag())])
        # ('clf_svm_bag', SVMBag(base_estimator=SVC)),
        # ('clf_adaboost', AdaBoostEnsemble()),
        # ('clf_adatree', AdaTree(base_estimator=DecisionTreeClassifier)),
        # ('clf_adabayes', AdaBayes()),
        # ('clf_adasvm', AdaSVM())])

    if args:
        voting_clf.set_params(**args)

    return voting_clf


def normalize(df_origin):
    """Fill missing values, drop unneeded columns and convert columns to appropriate dtypes"""
    df = df_origin.copy()
    drop_columns = ["name", "owner", "repository"]
    for c in drop_columns:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
    for c in df.columns:
        if df[c].dtype == 'O':
            if c in ['isOwnerHomepage', 'hasHomepage', 'hasLicense', 'hasTravisConfig', 'hasCircleConfig', 'hasCiConfig']:
                df[c] = (df[c] == 'True').astype(int)
            else:
                df[c].fillna('', inplace=True)
        else:
            df[c].fillna(0, inplace=True)
            df[c] = df[c].astype(int)
    return df


def keep_useful_features(df, useful_features):
    for c in df.columns:
        if c not in useful_features:
            df.drop(c, axis=1, inplace=True)
    for f in useful_features:
        if f not in df.columns:
            df[f] = 0
    return df
