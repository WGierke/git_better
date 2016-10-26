"""
Aggregate the provided Data Frame with features we receive from the Github REST and GraphQL API
GraphQL:    closed_issues, open_issues, closed_pull_requests, merged_pull_requests, open_pull_requests, projects, watchers,
            stargazers, forks, mentionableUsers, description
REST:       size, language_rate, readme
TODO:       commits_count
"""
from tqdm import tqdm
import re
from utils import request_graph_features
import json
from utils import get_client


def aggregate_features(data_frame):
    print "Loading features for " + str(len(data_frame)) + " repositories"
    for index, row in tqdm(data_frame.iterrows(), total=len(data_frame)):
        repo = get_client().get_repo(row['repository'])
        data_frame = add_graph_features(data_frame, index, row['repository'])
        data_frame = add_rest_features(data_frame, repo, index)
    return data_frame


def add_graph_features(data_frame, index, repo_path):
    features = get_graph_features(repo_path.split("/")[0], repo_path.split("/")[1])
    for k in features.keys():
        data_frame.set_value(index, k, features[k])
    return data_frame


def get_graph_features(repo_owner, repo_name):
    response = request_graph_features(repo_owner, repo_name)
    response = json.loads(response)
    data = response["data"]["repositoryOwner"]["repository"]
    features = {}
    for k in data.keys():
        if isinstance(data[k], dict):
            features[k] = data[k]['totalCount']
        else:
            features[k] = data[k]
    return features


def add_rest_features(data_frame, repo, index):
    data_frame.set_value(index, 'size', repo.size)
    # Readme
    regex = re.compile('[^a-zA-Z0-9 :\/]')
    readme = regex.sub('', repo.get_readme().decoded_content)
    data_frame.set_value(index, 'readme', readme)
    # Language_Rate
    sum_loc = sum(repo.get_languages().values())
    languages = repo.get_languages()
    for key in languages.keys():
        data_frame.set_value(index, "LANGUAGE_" + key, languages[key] / float(sum_loc))
    return data_frame
