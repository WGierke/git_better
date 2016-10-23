import os
import sys
import ConfigParser
from github import Github
import pandas

TRAINING_DATA_PATH = 'data/small_data.csv'


def load_github_client():
    """Initialize the Github client with the provided credentials"""
    if os.path.isfile("config.ini"):
        config = ConfigParser.RawConfigParser(allow_no_value=False)
        config.read('config.ini')
        GITHUB_USERNAME = config.get("github", "username")
        GITHUB_PASSWORD = config.get("github", "password")
    else:
        print "Please provide a config.ini file which includes your Github credentials"
        sys.exit(1)
    return Github(GITHUB_USERNAME, GITHUB_PASSWORD)


def load_training_data():
    "Read the training data"
    return pandas.read_csv(TRAINING_DATA_PATH)


def clean_data(data_frame):
    """Remove the https://github.com/ prefix"""
    data_frame['repository'] = data_frame['repository'].apply(lambda x: x.replace('https://github.com/', ''))
    return data_frame