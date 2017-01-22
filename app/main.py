from __future__ import division
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import argparse
import os
import pandas as pd
import sys
from tqdm import tqdm
from classifier import get_text_pipeline, get_voting_classifier, DescriptionClassifier, ReadmeClassifier, NumericEnsembleClassifier, normalize, EnsembleAllNumeric, keep_useful_features
from constants import VALIDATION_DATA_PATH, ADDITIONAL_VALIDATION_DATA_PATH
from evaluation import drop_text_features
from load_data import process_data
from preprocess import ColumnSumFilter, ColumnStdFilter, PolynomialTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from training import load_pickle, get_undersample_df, drop_defect_rows, JOBLIB_VOTING_PIPELINE_NAME, save_pickle

N_BEST_FEATURES = 100
NUMERIZE_README = False
SAVE_PCIKLES = True

def main():
    parser = argparse.ArgumentParser(
        description='Classify GitHub repositories using training data or a pre-trained model. The predicted repositories are saved in predictions.txt.')
    parser.add_argument('-i', '--input-file', required=True,
                        help='Path to the input file that should be classified e.g. "data/example-input.txt"')
    parser.add_argument('-t', '--training-file',
                        help='Path to the training file that should be used to train the classifier e.g. "data/example-output.txt". ' +
                        'Repository URL and label should be separated by a comma or a whitespace character.')
    parser.add_argument('-p', '--processed', action='store_true',
                        help='Specifies that training file already contains fetched features.')
    parser.add_argument('-l', '--loops',
                        help='Specifies how many classifiers should be trained on the given training data. The classifier with the highest average score on the validation data is used for the prediction (default=1).')
    args = parser.parse_args()

    if os.path.isfile(args.input_file):
        classify(args)
    else:
        print "The input file doesn't exist"
        sys.exit(1)


def classify(args):
    input_path = args.input_file
    df_input = get_input_data(input_path)
    print "Fetching features for {} input samples".format(len(df_input))
    df_input = process_data(data_frame=df_input)
    if args.training_file:
        if args.processed:
            df_train = pd.read_csv(args.training_file)
        else:
            df_train = pd.read_csv(args.training_file, sep=' ', names=[
                                   "repository", "label"])
            print "Fetching features for {} training samples".format(len(df_train))
            df_train = process_data(data_frame=df_train)
        loops = args.loops or 1
        loops = int(loops)
        train_and_predict(df_train, df_input, loops)
    else:
        predict(df_input)


def split_features(df_origin):
    """Split features in numeric features, description, readme and label"""
    df = df_origin.copy()
    df = drop_defect_rows(df)
    y = None
    if "label" in df.columns:
        y = df.pop("label")
    df = normalize(df)
    descr = df.pop("description")
    readme = df.pop("readme")
    return df, descr, readme, y


def train_and_predict(df_training, df_input, loops):
    """Use a VotingClassifier on top of an ensemble of numeric classifiers
    and classifiers for the description and readme features"""
    print 30 * "="
    print "Fitting {} Voting Classifier(s)".format(loops)
    print 30 * "="

    df_training = normalize(df_training)

    val_df = normalize(pd.read_csv(VALIDATION_DATA_PATH))
    val_df = keep_useful_features(val_df, df_training.columns)
    y_val = val_df.pop("label")
    val_add_df = normalize(pd.read_csv(ADDITIONAL_VALIDATION_DATA_PATH))
    val_add_df = keep_useful_features(val_add_df, df_training.columns)
    y_val_add = val_add_df.pop("label")

    _ = df_training.pop("Unnamed: 0")
    _ = val_df.pop("Unnamed: 0")
    _ = val_add_df.pop("Unnamed: 0")

    best_average_score = 0
    best_clf = None

    for i in tqdm(range(loops)):
        clf = VotingClassifier(estimators=[('description', DescriptionClassifier()),
                                                     ('readme', ReadmeClassifier()),
                                                     ('ensemble', NumericEnsembleClassifier())],
                               voting='soft')
        df_train = get_undersample_df(df_training.copy())
        _ = df_train.pop("index")
        y_train = df_train.pop("label")
        clf.fit(df_train, y_train)
        val_score = clf.score(val_df, y_val)
        val_add_score = clf.score(val_add_df, y_val_add)
        if (val_score + val_add_score) / 2 > best_average_score:
            best_clf = clf
            best_average_score = (val_score + val_add_score) / 2

    print 74 * "="
    print "Using trained Voting Classifier with average accuracy on validation sets of {0:2f}".format((best_average_score))
    print 74 * "="
    predict(df_input, model_voting=best_clf)

def predict(df_input, model_voting=None):
    if model_voting is None:
        print 35 * "="
        print 'Using pretrained Voting Classifier (67.74% on validation and 46.67% on additional validation data)'
        print 35 * "="
        model_voting = load_pickle(JOBLIB_VOTING_PIPELINE_NAME)
    repository = df_input["repository"]
    df_input = normalize(df_input)
    descr = df_input["description"]
    readme = df_input["readme"]
    df_input = keep_useful_features(df_input, model_voting.estimators_[-1].useful_features)
    df_input["description"] = descr
    df_input["readme"] = readme
    predictions = model_voting.predict(df_input)
    df_input["label"] = predictions
    df_input["repository"] = repository
    df_input["repository"] = df_input.repository.apply(lambda x: "https://github.com/" + x)
    df_input[["repository", "label"]].to_csv("predictions.txt", sep=' ', header=False, index=False, encoding='utf-8')
    print "Saved predictions in predictions.txt"
    return


def get_input_data(input_path):
    with open(input_path) as f:
        lines = f.read().splitlines()
    return pd.Series(lines).to_frame("repository")


def get_training_data(training_path):
    return pd.read_csv(training_path, sep=' ', names=["repository", "label"])


if __name__ == '__main__':
    main()
