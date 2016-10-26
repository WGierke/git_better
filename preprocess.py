import pandas
pandas.set_option('display.max_columns', 50)

TRAINING_DATA_PATH = 'data/small_data.csv'


def load_training_data():
    """Read the training data"""
    return pandas.read_csv(TRAINING_DATA_PATH)


def clean_data(data_frame):
    """Remove the https://github.com/ prefix"""
    data_frame['repository'] = data_frame['repository'].apply(lambda x: x.replace('https://github.com/', ''))
    return data_frame
