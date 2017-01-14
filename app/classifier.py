# code partly from our previous project Data Mining Cup, https://github.com/AlexImmer/run-dmc
import numpy as np, random
import logging
from scipy.sparse import csr_matrix
from scipy.stats import randint as sp_randint

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, \
    BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV

try:
    import theanets as tn
except Exception, e:
    logging.error("Can't import Theano: " + str(e))

try:
    import tensorflow as tf
    import tensorflow.contrib.learn as sklearn
except ImportError:
    print('Tensorflow not installed')

from xgboost import XGBClassifier
from operator import itemgetter


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
        self.clf.fit(df.values, Y)
        return self

    def predict(self, df):
        return self.clf.predict(df.values)

    def predict_proba(self, df):
        return self.clf.predict_proba(df.values)


    def get_params(self, **args):
        return self.clf.get_params(**args)


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
        self.clf = BernoulliNB(**args)


class SVM(GIClassifier):
    def __init__(self, **args):
        self.param_dist_random = {'shrinking': [True, False],
                                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                      'degree': sp_randint(2, 5)}
        self.clf = SVC(**args)



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


def get_voting_classifier():
    return VotingClassifier(voting='soft', estimators=[
        ('clf_bayes', NaiveBayes(binarize=False)),
        ('clf_tree', DecisionTree()),
        ('clf_forest', Forest()),
        ('clf_kneighbors', KNeighbors()),
        ('clf_svm', SVM(kernel='rbf', shrinking=True, probability=True)),
        ('clf_grad_boost', GradBoost()),
        ('clf_xgboost', XGBoost())])
        # ('clf_bag_ensemble', BagEnsemble()),
        #('clf_treebag', TreeBag())])
        # ('clf_svm_bag', SVMBag(base_estimator=SVC)),
        # ('clf_adaboost', AdaBoostEnsemble()),
        # ('clf_adatree', AdaTree(base_estimator=DecisionTreeClassifier)),
        # ('clf_adabayes', AdaBayes()),
        # ('clf_adasvm', AdaSVM())])
