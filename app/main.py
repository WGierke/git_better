from __future__ import division
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
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from training import load_pickle, get_undersample_df, drop_defect_rows, JOBLIB_DESCRIPTION_PIPELINE_NAME, JOBLIB_README_PIPELINE_NAME

N_BEST_FEATURES = 100
LOOPS = 10
NUMERIZE_README = False

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
        train_and_predict(df_train, df_input)
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


def train_and_predict(df_training, df_input):
    #use_numeric_flat_prediction(df_training.copy(), df_input.copy())
    use_mixed_stack_prediction(df_training.copy(), df_input.copy())


def use_numeric_flat_prediction(df, df_input):
    """Add the normalized term frequencies of the text features
       to the numeric features and train a set of classifiers on that."""
    print 50*"="
    print "Fitting Numeric Flat"
    print 50*"="
    df = normalize(df)

    val_df = normalize(pd.read_csv("data/validation_data_processed.csv"))
    val_add_df = normalize(pd.read_csv("data/validation_additional_processed_data.csv"))

    if not NUMERIZE_README:
        _ = df.pop("readme")
        _ = val_df.pop("readme")
        _ = val_add_df.pop("readme")

    y = df.pop("label")
    y_val = val_df.pop("label")
    y_val_add = val_add_df.pop("label")

    _ = df.pop("Unnamed: 0")
    _ = val_df.pop("Unnamed: 0")
    _ = val_add_df.pop("Unnamed: 0")

    ensemble_clf = EnsembleAllNumeric(n_jobs=-1).fit(df, y)
    df = ensemble_clf.transform(df)
    useful_features = ensemble_clf.useful_features
    val_df = ensemble_clf.transform(val_df)
    val_add_df = ensemble_clf.transform(val_add_df)

    model = ExtraTreesClassifier()
    model.fit(df, y)

    zipped = zip(useful_features, model.feature_importances_)
    zipped.sort(key=lambda x: x[1], reverse=True)
    best_features = [x[0] for x in zipped[:N_BEST_FEATURES]]

    df = keep_useful_features(df, best_features)
    val_df = keep_useful_features(val_df, best_features)
    val_add_df = keep_useful_features(val_add_df, best_features)
    df_input = keep_useful_features(df_input, best_features)

    df["label"] = y

    clfs = [clf[1] for clf in get_voting_classifier().estimators]
    clfs.append(ExtraTreesClassifier(n_jobs=-1))
    clfs.append(get_voting_classifier())

    val_scores = [0] *len(clfs)
    val_add_scores = [0] *len(clfs)

    for _ in tqdm(range(LOOPS)):
        df_train = get_undersample_df(df.copy())
        _ = df_train.pop("index")
        y_train = df_train.pop("label")
        for i in range(len(clfs)):
            try:
                clf = clfs[i]
                clf.fit(df_train, y_train)
                val_scores[i] += clf.score(val_df, y_val)
                val_add_scores[i] += clf.score(val_add_df, y_val_add)
            except Exception, e:
                print e
    for i in range(len(clfs)):
        print clfs[i].__class__
        print "Validation: " + str(val_scores[i]/LOOPS)
        print "Additional: " + str(val_add_scores[i]/LOOPS)


def use_mixed_stack_prediction(df_training, df_input):
    print 50*"="
    print "Fitting Mixed Stack"
    print 50*"="

    df_training = normalize(df_training)

    val_df = normalize(pd.read_csv(VALIDATION_DATA_PATH))
    val_df = keep_useful_features(val_df, df_training.columns)
    y_val = val_df.pop("label")
    val_df_add = normalize(pd.read_csv(ADDITIONAL_VALIDATION_DATA_PATH))
    val_df_add = keep_useful_features(val_df_add, df_training.columns)
    y_val_add = val_df_add.pop("label")

    _ = df_training.pop("Unnamed: 0")
    _ = val_df.pop("Unnamed: 0")
    _ = val_df_add.pop("Unnamed: 0")

    meta_ensemble = VotingClassifier(estimators=[('description', DescriptionClassifier()),
                                                 ('readme', ReadmeClassifier()),
                                                 ('ensemble', NumericEnsembleClassifier())],
                                    voting='soft')

    clfs = [DescriptionClassifier(), ReadmeClassifier(), NumericEnsembleClassifier(), meta_ensemble]
    clfs.extend([clf[1] for clf in get_voting_classifier().estimators])
    val_scores = [0] * len(clfs)
    val_add_scores = [0] * len(clfs)

    for _ in tqdm(range(LOOPS)):
        df_train = get_undersample_df(df_training.copy())
        _ = df_train.pop("index")
        y_train = df_train.pop("label")
        for i in range(len(clfs)):
            clf = clfs[i]
            clf.fit(df_train, y_train)
            val_scores[i] += clf.score(val_df, y_val)
            val_add_scores[i] += clf.score(val_df_add, y_val_add)
    for i in range(len(clfs)):
        print clfs[i].__class__
        print "Validation: " + str(val_scores[i]/LOOPS)
        print "Additional: " + str(val_add_scores[i]/LOOPS)


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
