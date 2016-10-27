from preprocess import load_training_data, clean_data
from feature_aggregation import aggregate_features
from utils import load_config

TRAINING_DATA_PATH = 'data/small_data.csv'


def process_data(training_data_path=TRAINING_DATA_PATH):
    load_config()
    data_frame = load_training_data(training_data_path)
    data_frame = clean_data(data_frame)
    data_frame = aggregate_features(data_frame)
    return data_frame

if __name__ == '__main__':
    process_data()
