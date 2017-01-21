from __future__ import division
import argparse
import os
import pandas as pd
import sys
from classifier import get_text_pipeline, get_voting_classifier, DescriptionClassifier, ReadmeClassifier, NumericEnsembleClassifier, normalize, EnsembleAllNumeric
from constants import VALIDATION_DATA_PATH, ADDITIONAL_VALIDATION_DATA_PATH
from evaluation import drop_text_features
from load_data import process_data
from preprocess import ColumnSumFilter, ColumnStdFilter, PolynomialTransformer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from training import load_pickle, get_undersample_df, drop_defect_rows, JOBLIB_DESCRIPTION_PIPELINE_NAME, JOBLIB_README_PIPELINE_NAME

N_BEST_FEATURES = 100

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
    use_numeric_flat_prediction(df_training.copy(), df_input.copy())
    use_mixed_stack_prediction(df_training.copy(), df_input.copy())


def use_numeric_flat_prediction(df, df_input):
    print 50*"="
    print "Fitting Numeric Flat"
    print 50*"="
    #df = get_undersample_df(df)
    df = normalize(df)

    val_df = normalize(pd.read_csv("data/validation_data_processed.csv"))
    val_add_df = normalize(pd.read_csv("data/validation_additional_processed_data.csv"))

    _ = df.pop("readme")
    _ = val_df.pop("readme")
    _ = val_add_df.pop("readme")

    y = df.pop("label")
    y_val = val_df.pop("label")
    y_val_add = val_add_df.pop("label")

    #_ = df.pop("Unnamed: 0")
    _ = df.pop("index")
    _ = val_df.pop("Unnamed: 0")
    _ = val_add_df.pop("Unnamed: 0")

    ensemble_clf = EnsembleAllNumeric(n_jobs=-1).fit(df, y)
    df = ensemble_clf.transform(df)
    useful_features = ensemble_clf.useful_features
    val_df = ensemble_clf.transform(val_df)
    val_df = ensemble_clf.keep_useful_features(val_df, useful_features)
    val_add_df = ensemble_clf.transform(val_add_df)
    val_add_df = ensemble_clf.keep_useful_features(val_add_df, useful_features)

    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(df, y)

    zipped = zip(useful_features, model.feature_importances_)
    zipped.sort(key=lambda x: x[1], reverse=True)
    best_features = [x[0] for x in zipped[:N_BEST_FEATURES]]

    df = ensemble_clf.keep_useful_features(df, best_features)
    val_df = ensemble_clf.keep_useful_features(val_df, best_features)
    val_add_df = ensemble_clf.keep_useful_features(val_add_df, best_features)
    df_input = ensemble_clf.keep_useful_features(df_input, best_features)

    df["label"] = y

    clfs = [clf[1] for clf in get_voting_classifier().estimators]
    clfs.append(ExtraTreesClassifier(n_jobs=-1))
    clfs.append(get_voting_classifier())

    loops = 10
    for clf in clfs:
        print clf.__class__
        val_score = 0
        val_add_score = 0
        for i in range(loops):
            X_train, X_test = train_test_split(df, test_size=0.3)
            y_train = X_train.pop("label")
            y_test = X_test.pop("label")

            clf.fit(X_train, y_train)
            val_score += clf.score(val_df, y_val)
            val_add_score += clf.score(val_add_df, y_val_add)
        print "Validation: " + str(val_score/loops)
        print "Additional: " + str(val_add_score/loops)
        # model = model.fit(X_train, y_train)
        # for set_name, X, y in [("Validation", val_df, y_val), ("Additional Validation", val_add_df, y_val_add)]:
        #     print "Score on {}: {}".format(set_name, model.score(X, y))
        # print "Prediction for input data:"
        # print model.predict(df_input)


def use_mixed_stack_prediction(df_training, df_input):
    print 50*"="
    print "Fitting Mixed Stack"
    print 50*"="

    df_val = pd.read_csv(VALIDATION_DATA_PATH)
    y_val = df_val.pop("label")
    df_val_add = pd.read_csv(ADDITIONAL_VALIDATION_DATA_PATH)
    y_val_add = df_val_add.pop("label")

    meta_ensemble = VotingClassifier(estimators=[('description', DescriptionClassifier()),
                                                 ('readme', ReadmeClassifier()),
                                                 ('ensemble', NumericEnsembleClassifier())],
                                    voting='soft')

    loops = 10
    clfs = [DescriptionClassifier(), ReadmeClassifier(), NumericEnsembleClassifier(), meta_ensemble]
    clfs.extend([clf[1] for clf in get_voting_classifier().estimators])
    for clf in clfs:
        print clf.__class__
        val_score = 0
        val_add_score = 0
        for i in range(loops):
            X_train, X_test = train_test_split(df_training, test_size=0.3)
            y_train = X_train.pop("label")
            y_test = X_test.pop("label")
            clf = clf.fit(X_train, y_train)
            val_score += clf.score(df_val, y_val)
            val_add_score += clf.score(df_val_add, y_val_add)
        print "Validation: " + str(val_score/loops)
        print "Additional: " + str(val_add_score/loops)
            # for set_name, X, y in [("Test", X_test, y_test), ("Validation", df_val, y_val), ("Additional Validation", df_val_add, y_val_add)]:
            #     print "Score on {}: {}".format(set_name, model.score(X, y))
            # print "Prediction for input data:"
            # print model.predict(df_input)


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
