from preprocess import load_training_data, clean_data
from feature_aggregation import aggregate_features
from utils import load_config

if __name__ == '__main__':
    load_config()
    data_frame = load_training_data()
    data_frame = clean_data(data_frame)
    data_frame = aggregate_features(data_frame)
