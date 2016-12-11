import json
import os
import Queue
import threading
from feature_aggregation import aggregate_features
from preprocess import load_training_data, clean_data
from crawl import write_label_links
from tqdm import tqdm
import pandas as pd
from utils import load_config
from constants import TRAINING_DATA_PATH, PROCESSED_DATA_PATH
from training import find_best_repository_classification
from app.evaluation import get_cleaned_processed_df, drop_text_features


def load_features_async(data_frame):
    """Put the data frame in a thread safe queue, spawn 5 worker threads and aggregate the features asynchronously"""
    THREADS_COUNT = 10
    bar = tqdm(total=len(data_frame))
    df_q = Queue.LifoQueue()
    df_q.put(data_frame)
    token_q = Queue.LifoQueue()
    for i in range(THREADS_COUNT):
        token_q.put(i)

    threads = []
    for index, row in data_frame.iterrows():
        t = threading.Thread(target=aggregate_features, args=(index, row, bar, df_q, token_q))
        t.daemon = True
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
    return df_q.get()


def process_data(training_data_path=TRAINING_DATA_PATH, data_frame=None):
    """Load data training data from the provided path or take the given data frame and add all features"""
    if data_frame is None:
        data_frame = load_training_data(training_data_path)

    load_config()
    data_frame = clean_data(data_frame)
    data_frame = load_features_async(data_frame)
    data_frame["description"].fillna("", inplace=True)
    data_frame["readme"].fillna("", inplace=True)
    data_frame.fillna(0, inplace=True)
    return data_frame


def process_links(file_path="data/docs_links.txt", label=None):
    """Takes a file of repository links and returns the processed data frame.
    Only the first 170 links are processed due to API limit rates."""
    n = 170
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            content = f.read()
    else:
        print "File does not exist"

    links = list(json.loads(content))
    if not label:
        label = file_path.split("data/")[1].split("_links")[0]
    label = label.upper()

    df = pd.DataFrame([[l, label] for l in links], columns=["repository", "label"])
    df = df[:n]
    df = process_data(data_frame=df)

    for link in df["repository"]:
        links.remove("https://github.com/" + link)
    write_label_links(links, label=label, path=file_path)
    return df


if __name__ == '__main__':
    data_frame = get_cleaned_processed_df()
    #data_frame.to_csv("tsv.tsv", sep="\t")
    data_frame = drop_text_features(data_frame)
    #data_frame.to_csv(PROCESSED_DATA_PATH, encoding='utf-8')
    y_train = data_frame.pop("label")
    find_best_repository_classification(data_frame, y_train)
