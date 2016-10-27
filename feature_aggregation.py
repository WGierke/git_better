"""
Aggregate the provided Data Frame with features we receive from the Github REST and GraphQL API
GraphQL:    closed_issues, open_issues, closed_pull_requests, merged_pull_requests, open_pull_requests, projects, watchers,
            stargazers, forks, mentionableUsers, description
REST:       size, language_rate, readme
Custom:     isOwnerHomepage, hasHomepage
TODO:       commits_count
"""
from tqdm import tqdm
import re
from utils import request_graph_features, website_exists
import json
from utils import get_client


def aggregate_features(data_frame):
    print "Loading features for " + str(len(data_frame)) + " repositories"
    for index, row in tqdm(data_frame.iterrows(), total=len(data_frame)):
        repo = get_client().get_repo(row['repository'])
        owner = data_frame['owner'][index]
        name = data_frame['name'][index]
        data_frame = add_graph_features(data_frame, index, owner, name)
        data_frame = add_rest_features(data_frame, index, repo)
        data_frame = add_custom_features(data_frame, index, owner, name)
    return data_frame


def add_graph_features(data_frame, index, owner, name):
    features = get_graph_features(owner, name)
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


def add_rest_features(data_frame, index, repo):
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


def add_custom_features(data_frame, index, owner, name):
    data_frame.set_value(index, "isOwnerHomepage", name == "{}.github.io".format(owner))
    data_frame.set_value(index, "hasHomepage", website_exists("http://{}.github.io/{}".format(owner, name)))
    return data_frame
