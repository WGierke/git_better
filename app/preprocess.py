import pandas


def load_training_data(training_data_path):
    """Read the training data"""
    return pandas.read_csv(training_data_path)


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
