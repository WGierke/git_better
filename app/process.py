from feature_aggregation import aggregate_features
from preprocess import load_training_data, clean_data
from tqdm import tqdm
from utils import load_config
from constants import TRAINING_DATA_PATH, PROCESSED_DATA_PATH
import Queue
import threading


def load_features_async(data_frame):
    """Put the data frame in a thread safe queue, spawn as many threads as
    there are rows and aggregate the features asynchronously"""
    bar = tqdm(total=len(data_frame))
    q = Queue.LifoQueue()
    q.put(data_frame)

    threads = []
    for index, row in data_frame.iterrows():
        t = threading.Thread(target=aggregate_features, args=(index, row, bar, q))
        t.daemon = True
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
    return q.get()


def process_data(training_data_path=TRAINING_DATA_PATH):
    load_config()
    data_frame = load_training_data(training_data_path)
    data_frame = clean_data(data_frame)
    data_frame = load_features_async(data_frame)
    data_frame["description"].fillna("", inplace=True)
    data_frame["readme"].fillna("", inplace=True)
    data_frame.fillna(0, inplace=True)
    return data_frame


if __name__ == '__main__':
    data_frame = process_data()
    data_frame.to_csv(PROCESSED_DATA_PATH, encoding='utf-8')
