from preprocess import load_github_client, load_training_data, clean_data
from feature_aggregation import aggregate_features

if __name__ == '__main__':
    github_client = load_github_client()
    data_frame = load_training_data()
    data_frame = clean_data(data_frame)
    data_frame = aggregate_features(github_client, data_frame)
