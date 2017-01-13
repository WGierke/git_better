from __future__ import division
import argparse
import os
import sys
import pandas as pd
import numpy as np
from load_data import process_data
from classifier import get_numeric_ensemble
from training import load_pickle, get_text_pipeline, \
    get_undersample_df, drop_defect_rows, \
    JOBLIB_DESCRIPTION_PIPELINE_NAME, JOBLIB_README_PIPELINE_NAME
from evaluation import complete_columns, drop_text_features
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.cross_validation import train_test_split
from preprocess import ColumnSumFilter, ColumnStdFilter, PolynomialTransformer
from sklearn.pipeline import Pipeline


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
    args = parser.parse_args()

    if os.path.isfile(args.input_file):
        classify(args)
    else:
        print "The input file doesn't exist"
        sys.exit(1)


def classify(args):
    input_path = args.input_file
    df_input = get_input_data(input_path)
    df_input = process_data(data_frame=df_input)
    if args.training_file:
        if args.processed:
            df_train = pd.read_csv(args.training_file)
        else:
            df_train = pd.read_csv(args.training_file, sep=' ', names=["repository", "label"])
            df_train = process_data(data_frame=df_train)
        #df_train = drop_defect_rows(df_train)
        #df_train = get_undersample_df(df_train)
        train_and_predict(df_train, df_input)
    else:
        predict(df_input)


def train_and_predict(df_train, df_X):
    df_val = pd.read_csv("data/validation_data.csv")
    df_train = drop_text_features(df_train)
    df_X = drop_text_features(df_X)
    df_val = drop_text_features(df_val)
    y_val = df_val.pop("label")
    y_train = df_train.pop("label")

    df_train.fillna(0, inplace=True)
    df_X.fillna(0, inplace=True)
    df_val.fillna(0, inplace=True)

    for df in [df_val, df_train, df_X]:
        for c in df.columns:
            df[c] = df[c].astype(int)

    ppl = Pipeline([
        ('clmn_std_filter', ColumnStdFilter(min_std=1)),
        ('clmn_sum_filter', ColumnSumFilter(min_sum=10)),
    ])

    ppl = ppl.fit(df_train)
    df_train = ppl.fit_transform(df_train)
    useful_features = list(df_train.columns)

    #poly_transf = PolynomialTransformer(degree=2)
    #df_train = poly_transf.transform(df_train)

    #Reset indices
    df_train["label"] = y_train
    print useful_features
    df_X = keep_useful_features(useful_features, df_X)
    df_val = keep_useful_features(useful_features, df_val)

    #df_X = poly_transf.transform(df_X)
    #df_val = poly_transf.transform(df_val)

    X_train, X_test = train_test_split(df_train, test_size=0.3)
    y_train = X_train.pop("label")
    y_test = X_test.pop("label")
    ensemble_numeric = get_numeric_ensemble().fit(X_train, y_train)
    print "Score on Test set: " + str(ensemble_numeric.score(X_test, y_test))
    print "Score on Validation set: " + str(ensemble_numeric.score(df_val, y_val))
    import ipdb; ipdb.set_trace()
    print "Prediction for input: " + str(ensemble_numeric.predict(df_X))


def keep_useful_features(useful_features, df):
    for c in df.columns:
        if c not in useful_features:
            df.drop(c, axis=1, inplace=True)
    for f in useful_features:
        if f not in df.columns:
            df[f] = 0
    return df


def predict(df_input):
    model_description = load_pickle(JOBLIB_DESCRIPTION_PIPELINE_NAME)
    model_readme = load_pickle(JOBLIB_README_PIPELINE_NAME)
    probabilities = [model_description.predict_proba(df_input["description"])]
    probabilities.append(model_readme.predict_proba(df_input["readme"]))
    probabilities = [sum(e)/len(e) for e in zip(*probabilities)]
    predictions = [model_readme.classes_[list(prob).index(max(prob))] for prob in probabilities]
    labels = predictions
    df_input["label"] = labels
    df_input.repository = df_input.repository.apply(lambda x: "https://github.com/" + x)
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
