from __future__ import division
import argparse
import os
import sys
import pandas as pd
from load_data import process_data
from classifier import get_voting_classifier
from training import load_pickle, get_text_pipeline, \
    get_undersample_df, drop_defect_rows, \
    JOBLIB_DESCRIPTION_PIPELINE_NAME, JOBLIB_README_PIPELINE_NAME
from constants import VALIDATION_DATA_PATH, ADDITIONAL_VALIDATION_DATA_PATH
from evaluation import drop_text_features
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
            df_train = pd.read_csv(args.training_file, sep=' ', names=[
                                   "repository", "label"])
            df_train = process_data(data_frame=df_train)
        df_train = get_undersample_df(df_train)
        train_and_predict(df_train, df_input)
    else:
        predict(df_input)


def normalize(df_origin):
    """Fill missing values, drop unneeded columns and convert columns to appropriate dtypes"""
    df = df_origin.copy()
    df.drop(["name", "owner", "repository"], axis=1, inplace=True)
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


def train_and_predict(df_training, df_input):
    df_val = pd.read_csv(VALIDATION_DATA_PATH)
    df_val_add = pd.read_csv(ADDITIONAL_VALIDATION_DATA_PATH)

    X_training, train_descr, train_readme, y_train = split_features(df_training)
    X_input, input_descr, input_readme, _ = split_features(df_input)
    X_val, val_descr, val_readme, y_val = split_features(df_val)
    X_val_add, val_add_descr, val_add_readme, y_val_add = split_features(df_val_add)

    ppl_text_descr = get_text_pipeline()
    ppl_text_descr = ppl_text_descr.fit(train_descr, y_train)
    print "Val Descr Score: " + str(ppl_text_descr.score(val_descr, y_val))
    print "Val Add Descr Score: " + str(ppl_text_descr.score(val_add_descr, y_val_add))
    print ppl_text_descr.predict(input_descr)

    ppl_text_readme = ppl_text_descr.fit(train_readme, y_train)
    print "Val Readme Score: " + str(ppl_text_readme.score(val_readme, y_val))
    print "Val Add Readme Score: " + str(ppl_text_readme.score(val_add_readme, y_val_add))
    print ppl_text_readme.predict(input_readme)

    ppl = Pipeline([
        ('clmn_std_filter', ColumnStdFilter(min_std=1)),
        ('clmn_sum_filter', ColumnSumFilter(min_sum=10)),
    ])

    ppl = ppl.fit(X_training)
    X_training = ppl.transform(X_training)
    useful_features = list(X_training.columns)

    X_input = keep_useful_features(useful_features, X_input)
    X_val = keep_useful_features(useful_features, X_val)
    X_val_add = keep_useful_features(useful_features, X_val_add)

    poly_transf = PolynomialTransformer(degree=2)
    X_training = poly_transf.transform(X_training)
    X_input = poly_transf.transform(X_input)
    X_val = poly_transf.transform(X_val)
    X_val_add = poly_transf.transform(X_val_add)

    X_training["label"] = y_train

    X_train, X_test = train_test_split(X_training, test_size=0.3)
    y_train = X_train.pop("label")
    y_test = X_test.pop("label")
    ensemble_numeric = get_voting_classifier().fit(X_train, y_train)

    print "Score on Test set: " + str(ensemble_numeric.score(X_test, y_test))
    print "Score on Validation set: " + str(ensemble_numeric.score(X_val, y_val))
    print "Score on Additional Validation set: " + str(ensemble_numeric.score(X_val_add, y_val_add))
    print "Prediction for input: " + str(ensemble_numeric.predict(X_input))


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
    probabilities = [sum(e) / len(e) for e in zip(*probabilities)]
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
