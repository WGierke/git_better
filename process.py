from preprocess import load_training_data, clean_data
from feature_aggregation import aggregate_features
from utils import load_config
import threading
from tqdm import tqdm

TRAINING_DATA_PATH = 'data/training_data.csv'
PROCESSED_DATA_PATH = 'data/processed_data.csv'


def load_features_async(data_frame):
    bar = tqdm(total=len(data_frame))
    lock = threading.Lock()

    threads = []
    print "Loading features for " + str(len(data_frame)) + " repositories"
    for index, row in data_frame.iterrows():
        t = threading.Thread(target=aggregate_features, args=(data_frame, index, row, bar, lock))
        t.daemon = True
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
    return data_frame


def process_data(training_data_path=TRAINING_DATA_PATH):
    load_config()
    data_frame = load_training_data(training_data_path)
    data_frame = clean_data(data_frame)
    data_frame = load_features_async(data_frame)
    return data_frame


if __name__ == '__main__':
    data_frame = process_data()
    data_frame.to_csv(PROCESSED_DATA_PATH, encoding='utf-8')
