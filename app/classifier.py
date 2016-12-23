# code partly from our previous project Data Mining Cup, https://github.com/AlexImmer/run-dmc
import numpy as np, random
import logging
from scipy.sparse import csr_matrix
from scipy.stats import randint as sp_randint

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, \
    BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    import theanets as tn
except Exception, e:
    logging.error("Can't import Theano: " + str(e))

try:
    import tensorflow.contrib.learn as skflow
except ImportError:
    print('Tensorflow not installed')

from xgboost import XGBClassifier
from operator import itemgetter


class GIClassifier(object):
    clf = None

    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        assert len(Y) == X.shape[0]
        self.X = X
        self.Y = Y
        self.tune_parameters = tune_parameters
        self.random_search = random_search

    def __call__(self):
        if self.tune_parameters and self.is_searchable():
            if self.random_search:
                print(self.clf.get_params().keys())
                try:
                    return self.estimate_parameters_with_random_search()
                except Exception as e:
                    print(e)
                    pass

            else:
                print(self.clf.get_params().keys())
                try:
                    return self.estimate_parameters_with_grid_search()
                except Exception as e:
                    print(e)
                    pass
        else:
            return self.fit()

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
                                           n_iter=30, n_jobs=-1)
        random_search.fit(self.X, self.Y)
        print("Random Search")
        #self.report(random_search.grid_scores_)
        print("best estimator, params and score: ")
        print(random_search.best_estimator_)
        print(random_search.best_params_)
        print(random_search.best_score_)
        return random_search

    def estimate_parameters_with_grid_search(self):
        grid_search = GridSearchCV(self.clf, param_grid=self.param_grid,n_jobs=-1)
        grid_search.fit(self.X, self.Y)
        print("Grid Search")
        #print("cv_results_: ")
        #print(grid_search.cv_results_)
        #print("report: ")
        #self.report(grid_search.cv_results_)
        print("best estimator, params and score: ")
        print(grid_search.best_estimator_)
        print(grid_search.best_params_)
        print(grid_search.best_score_)
        return grid_search

    def fit(self):
        self.clf.fit(self.X, self.Y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def is_searchable(self):
        return True


class DecisionTree(GIClassifier):
    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(DecisionTree, self).__init__(X, Y, tune_parameters, random_search)
        if tune_parameters:
            self.param_dist_random = {'max_depth': sp_randint(1, 100),
                                      'min_samples_leaf': sp_randint(1, 150),
                                      'max_features': sp_randint(1, self.X.shape[1] - 1),
                                      'criterion': ['entropy', 'gini']}
            self.param_grid = {'max_depth': (1,3,10,30,50,100),
                               'criterion': ('entropy', 'gini')}
        self.clf = DecisionTreeClassifier()


class Forest(GIClassifier):
    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(Forest, self).__init__(X, Y, tune_parameters, random_search)
        if tune_parameters:
            self.param_dist_random = {'max_depth': sp_randint(1, 100),
                                      'min_samples_leaf': sp_randint(1, 100),
                                      'max_features': sp_randint(1, self.X.shape[1] - 1),
                                      'criterion': ['entropy', 'gini']}
        self.clf = RandomForestClassifier(n_estimators=100, n_jobs=8)


class KNeighbors(GIClassifier):
    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(KNeighbors, self).__init__(X, Y, tune_parameters, random_search)
        if tune_parameters:
            self.param_dist_random = {'leaf_size':sp_randint(20, 50),
                                    'n_neighbors': sp_randint(4, 30)}
        self.clf = KNeighborsClassifier()


class NaiveBayes(GIClassifier):
    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(NaiveBayes, self).__init__(X, Y, tune_parameters, random_search)
        if tune_parameters:
            self.param_dist_random = {'alpha':sp_randint(0, 1)}
        self.clf = BernoulliNB(binarize=True)


class SVM(GIClassifier):
    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(SVM, self).__init__(X, Y, tune_parameters, random_search)
        if tune_parameters:
            self.param_dist_random = {'shrinking': [True, False],
                                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                      'degree': sp_randint(2, 5)}
        self.clf = SVC(kernel='rbf', shrinking=True)

    def predict_proba(self, X):
        return self.clf.decision_function(X)


class BagEnsemble(GIClassifier):
    classifier = None
    estimators = 20
    max_features = .5
    max_samples = .5

    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(BagEnsemble, self).__init__(X, Y, tune_parameters, random_search)
        if tune_parameters:
            self.param_dist_random = {'max_features': sp_randint(1, self.X.shape[1]),
                                      'n_estimators': sp_randint(1, 100)}
        self.clf = BaggingClassifier(self.classifier, n_estimators=self.estimators, n_jobs=8,
                                     max_samples=self.max_samples, max_features=self.max_features)


class TreeBag(BagEnsemble):
    classifier = DecisionTreeClassifier()

    def is_searchable(self):
        return False


class SVMBag(GIClassifier):
    classifier = None
    estimators = 10
    max_features = .5
    max_samples = .5

    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(SVMBag, self).__init__(X, Y, tune_parameters, random_search)
        self.X, self.Y = X, Y
        self.classifier = SVC(decision_function_shape='ovo')
        self.clf = BaggingClassifier(self.classifier, n_estimators=self.estimators, n_jobs=8,
                                     max_samples=self.max_samples, max_features=self.max_features)
    def is_searchable(self):
        return False

    def predict(self, X):
        X = X
        return self.clf.predict(X)


class AdaBoostEnsemble(GIClassifier):
    classifier = None
    estimators = 800
    learning_rate = .25
    algorithm = 'SAMME.R'

    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(AdaBoostEnsemble, self).__init__(X, Y, tune_parameters, random_search)
        if tune_parameters:
            self.param_dist_random = {'n_estimators': sp_randint(1, 1000),
                                      'algorithm': ['SAMME', 'SAMME.R'],
                                      'learning_rate': random.random(100)}
            self.param_dist_grid = {'n_estimators': [100, 200, 400, 900, 1000],
                                    'algorithm': ['SAMME', 'SAMME.R'],
                                    'learning_rate': [.1, .2, 0.25, .3,
                                                      .4, .5, .6]}
        self.clf = AdaBoostClassifier(self.classifier,
                                      n_estimators=self.estimators,
                                      learning_rate=self.learning_rate,
                                      algorithm=self.algorithm)


class AdaTree(AdaBoostEnsemble):
    classifier = DecisionTreeClassifier()


class AdaBayes(AdaBoostEnsemble):
    classifier = BernoulliNB()


class AdaSVM(AdaBoostEnsemble):
    algorithm = 'SAMME'

    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(AdaSVM, self).__init__(X, Y, tune_parameters, random_search)
        self.classifier = SVC(decision_function_shape='ovo')


class GradBoost(GIClassifier):
    estimators = 2000
    learning_rate = 1
    max_depth = 1
    max_features = 0.97

    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(GradBoost, self).__init__(X, Y, tune_parameters, random_search)
        self.clf = GradientBoostingClassifier(n_estimators=self.estimators,
                                              learning_rate=self.learning_rate,
                                              max_depth=self.max_depth,
                                              max_features=self.max_features)

    def is_searchable(self):
        return False

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict(X)


class XGBoost(GIClassifier):
    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(XGBoost, self).__init__(X, Y, tune_parameters, random_search)
        if tune_parameters:
            self.param_dist_random = {'max_depth': sp_randint(1, 20),
                                      'n_estimators' : sp_randint(50, 200)}
        self.clf = XGBClassifier()



class TheanoNeuralNetwork(GIClassifier):
   def __init__(self, X, Y, tune_parameters=False, random_search=False):
       super(TheanoNeuralNetwork, self).__init__(X, Y, tune_parameters, random_search)
       input_layer, output_layer = self.X.shape[1], len(np.unique(Y))
       inp = tn.layers.base.Input(size=input_layer, sparse='csr')
       self.clf = tn.Classifier(layers=[inp,
                                        (100, 'linear'), (50, 'norm:mean+relu'),
                                        output_layer])

   def is_searchable(self):
       return False

   def fit(self):
       self.clf.train((self.X, self.Y), algo='sgd', learning_rate=.05, momentum=0.9)
       return self


class TensorFlowNeuralNetwork(GIClassifier):
    steps = 20000
    learning_rate = 0.05
    hidden_units = [100, 100]
    optimizer = 'SGD'

    def __init__(self, X, Y, tune_parameters=False, random_search=False):
        super(TensorFlowNeuralNetwork, self).__init__(X, Y, tune_parameters, random_search)
        self.X = X.todense()  # TensorFlow/Skflow doesn't support sparse matrices
        output_layer = len(np.unique(Y))
        if tune_parameters:
            self.param_dist_random = {'learning_rate': random.random(100),
                                      'optimizer': ['Adam'],
                                      'hidden_units': [sp_randint(50, 500), sp_randint(50, 500)]}

        self.clf = skflow.TensorFlowDNNClassifier(hidden_units=self.hidden_units,
                                                  n_classes=output_layer, steps=self.steps,
                                                  learning_rate=self.learning_rate, verbose=0,
                                                  optimizer=self.optimizer)

    def predict(self, X):
        X = X.todense()  # TensorFlow/Skflow doesn't support sparse matrices
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X.todense())
