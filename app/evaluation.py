import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import PROCESSED_DATA_PATH, VALIDATION_DATA_PATH
from sklearn.metrics import accuracy_score, confusion_matrix


def eval_classifier(clf, X, y_correct, classes, plot_cm=True):
    """Given a classifier, the unlabeled data, the labels and the existing classes predict the corresponsing labels
       and return the accuracy"""
    y_pred = clf.predict(X)
    return get_accuracy_and_plot_confusion(y_correct, y_pred, classes, plot=plot_cm)


def get_accuracy_and_plot_confusion(y_correct, y_pred, classes, plot=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """Return the accuracy of the prediction and plot the corresponding confusion matrix if desired"""
    if plot:
        cm = confusion_matrix(y_correct, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    return accuracy_score(y_correct, y_pred)


def complete_columns(training_df, valid_df):
    """It's possible that training repositories have different programming languages than the repos in the validation set.
       Thus, both data frames have different feature columns. This method completes the columns for both data frames."""
    for c in valid_df.columns:
        if c not in training_df.columns:
            training_df[c] = 0
    for c in training_df.columns:
        if c not in valid_df.columns:
            valid_df[c] = 0
    return training_df, valid_df


def drop_text_features(df):
    df.drop('readme', axis=1, inplace=True)
    df.drop('description', axis=1, inplace=True)
    df.drop('repository', axis=1, inplace=True)
    df.drop('owner', axis=1, inplace=True)
    df.drop('name', axis=1, inplace=True)
    return df


def get_training_and_validation_df():
    """Returns X_train, y_train, X_valid, y_valid"""
    df = pd.DataFrame.from_csv(PROCESSED_DATA_PATH)
    val_df = pd.DataFrame.from_csv(VALIDATION_DATA_PATH)
    y_train = df.pop("label")
    y_val = val_df.pop("label")

    df, val_df = complete_columns(df, val_df)
    df.fillna(0, inplace=True)
    val_df.fillna(0, inplace=True)

    df = drop_text_features(df)
    val_df = drop_text_features(val_df)
    return df.values, y_train, val_df.values, y_val