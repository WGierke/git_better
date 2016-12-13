import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


def load_training_data(training_data_path):
    """Read the training data"""
    return pd.read_csv(training_data_path)


def clean_data(data_frame):
    data_frame = remove_prefix(data_frame)
    data_frame = split_name(data_frame)
    return data_frame


def remove_prefix(data_frame):
    """Remove the https://github.com/ prefix"""
    data_frame['repository'] = data_frame['repository'].apply(lambda x: x.replace('https://github.com/', ''))
    return data_frame


def split_name(data_frame):
    data_frame['owner'] = data_frame['repository'].apply(lambda x: x.split("/")[0])
    data_frame['name'] = data_frame['repository'].apply(lambda x: x.split("/")[1])
    return data_frame


def get_text_feature_names(df):
    return list(df.select_dtypes(include=['object']).columns)


class ColumnSumFilter(TransformerMixin):
    """Drop columns whose sum is smaller than min_sum"""

    def __init__(self, min_sum=-1):
        self.min_sum = min_sum

    def transform(self, X):
        number_columns = X.select_dtypes(include=['number']).columns
        transformed_X = X.copy()
        for c in number_columns:
            if transformed_X[c].sum() < self.min_sum:
                transformed_X.drop(c, axis=1, inplace=True)
        return transformed_X

    def fit(self, X, y=None, **fit_params):
        return self


class ColumnStdFilter(TransformerMixin):
    """Drop columns whose standard deviation is smaller than min_std"""

    def __init__(self, min_std=-1):
        self.min_std = min_std

    def transform(self, X):
        number_columns = X.select_dtypes(include=['number']).columns
        transformed_X = X.copy()
        for c in number_columns:
            if transformed_X[c].std() < self.min_std:
                transformed_X.drop(c, axis=1, inplace=True)
        return transformed_X

    def fit(self, X, y=None, **fit_params):
        return self
