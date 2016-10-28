from preprocess import load_training_data, clean_data
from feature_aggregation import aggregate_features
from utils import load_config

TRAINING_DATA_PATH = 'data/training_data.csv'
PROCESSED_DATA_PATH = 'data/processed_data.csv'


def process_data(training_data_path=TRAINING_DATA_PATH):
    load_config()
    data_frame = load_training_data(training_data_path)
    data_frame = clean_data(data_frame)
    data_frame = aggregate_features(data_frame)
    return data_frame


if __name__ == '__main__':
    data_frame = process_data()
    data_frame.to_csv(PROCESSED_DATA_PATH, encoding='utf-8')
