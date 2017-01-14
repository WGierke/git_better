"""
Aggregate the provided Data Frame with features we receive from the Github REST and GraphQL API
TODO:       check folder names
"""
from utils import get_client, get_last_repos_pagination_page
from utils import request_graph_features, website_exists
import traceback
import json
import re
import requests
import pandas as pd

REPO_API_URL = "https://api.github.com/repos/"


def aggregate_features(index, row, bar, df_q, token_q):
    repo = get_client().get_repo(row['repository'])
    owner = row['owner']
    name = row['name']

    token = token_q.get()
    try:
        new_data_frame = pd.DataFrame.from_dict(row).T
        new_data_frame = add_graph_features(new_data_frame, index, owner, name)
        new_data_frame = add_rest_features(new_data_frame, index, repo)
        new_data_frame = add_custom_features(new_data_frame, index, owner, name)
        if "closed_issues" in new_data_frame.columns:
            new_data_frame = fix_closed_issues(new_data_frame, index)
    except Exception, e:
        print "Exception in aggregate_features: " + str(e)
        token_q.put(token)
        return

    token_q.put(token)
    shared_data_frame = df_q.get()
    update_columns = [col for col in new_data_frame.columns if col not in ['repository', 'owner', 'name', 'label']]
    for col in update_columns:
        try:
            shared_data_frame.set_value(index, col, new_data_frame.loc[index, col])
        except Exception, e:
            print "An error occured while fetching {}/{} and setting {}: {}".format(owner, name, col, e)
    df_q.put(shared_data_frame)
    bar.update()


def add_graph_features(data_frame, index, owner, name):
    """closed_issues, open_issues, closed_pull_requests, merged_pull_requests, open_pull_requests,
    projects, watchers, stargazers, forks, mentionableUsers, description"""
    features = get_graph_features(data_frame, index, owner, name)
    for k in features.keys():
        data_frame.set_value(index, k, features[k])
    return data_frame


def get_graph_features_of_renamed_repo(data_frame, index, repo_owner, repo_name):
    new_repo_name = "new_repo_name"
    try:
        r = requests.get("https://github.com/{}/{}".format(repo_owner, repo_name))
        if len(r.history) < 1:
            return {} # repo doesn't exist at all
        new_repo_link = r.history[0].headers['location']
        new_repo_link = new_repo_link.replace('https://github.com/', '')
        new_repo_owner = new_repo_link.split("/")[0]
        new_repo_name = new_repo_link.split("/")[1]
        return get_graph_features(data_frame, index, new_repo_owner, new_repo_name)
    except Exception, e:
        print "Error while requesting renamed repo {}:{}".format(new_repo_name, e)
        return {}


def get_graph_features(data_frame, index, repo_owner, repo_name):
    response = request_graph_features(repo_owner, repo_name)
    response = json.loads(response)

    if "data" in response.keys():
        data = None
        try:
            data = response["data"]["repositoryOwner"]["repository"]
        except:
            print "Couldn't extract graph features data from " + str(response["data"])
        features = {}
        if data is None:
            return get_graph_features_of_renamed_repo(data_frame, index, repo_owner, repo_name)
        for k in data.keys():
            if isinstance(data[k], dict):
                features[k] = data[k]['totalCount']
            else:
                if data[k]:
                    features[k] = data[k]
                else:
                    features[k] = ''
        return features
    elif "message" in response.keys():
        print "An error occured while fetching GraphQL API: " + response["message"]
        raise Exception('Graph QL Error: ' + response['message'])
    else:
        print "An error occured while fetching GraphQL API: " + response
        raise Exception('Graph QL Error: ' + response['message'])
    return {}


def add_rest_features(data_frame, index, repo):
    """size, language_rate, readme"""
    data_frame.set_value(index, 'size', repo.size)
    try:
        readme = repo.get_readme().decoded_content
        data_frame.set_value(index, 'readme', readme)
    except:
        data_frame.set_value(index, 'readme', '')
    # Language_Rate
    sum_loc = sum(repo.get_languages().values())
    languages = repo.get_languages()
    for key in languages.keys():
        data_frame.set_value(index, "LANGUAGE_" + key,
                             languages[key] / float(sum_loc))
    return data_frame


def add_custom_features(data_frame, index, owner, name):
    """isOwnerHomepage, hasHomepage, hasLicense, hasTravisConfig, hasCircleConfig,
    hasCiConfig, commitsCount, branchesCount, tagsCount, releasesCount"""
    is_owner_homepage = name.lower() == "{}.github.io".format(owner.lower()) or name.lower() == "{}.github.com".format(owner.lower())
    has_homepage = website_exists("http://{}.github.io/{}".format(owner, name))
    has_license = "octicon octicon-law" in requests.get("https://github.com/{}/{}".format(owner, name)).text
    has_travis_config = website_exists("https://github.com/{}/{}/blob/master/.travis.yml".format(owner, name), only_headers=True)
    has_circle_config = website_exists("https://github.com/{}/{}/blob/master/circle.yml".format(owner, name), only_headers=True)
    has_ci_config = has_travis_config or has_circle_config
    commits_count = get_last_repos_pagination_page("{}/{}/commits?per_page=1".format(owner, name))
    branches_count = get_last_repos_pagination_page("{}/{}/branches?per_page=1".format(owner, name))
    tags_count = get_last_repos_pagination_page("{}/{}/tags?per_page=1".format(owner, name))
    releases_count = get_last_repos_pagination_page("{}/{}/releases?per_page=1".format(owner, name))

    data_frame.set_value(index, "isOwnerHomepage", is_owner_homepage)
    data_frame.set_value(index, "hasHomepage", has_homepage)
    data_frame.set_value(index, "hasLicense", has_license)
    data_frame.set_value(index, "hasTravisConfig", has_travis_config)
    data_frame.set_value(index, "hasCircleConfig", has_circle_config)
    data_frame.set_value(index, "hasCiConfig", has_ci_config)
    data_frame.set_value(index, "commitsCount", commits_count)
    data_frame.set_value(index, "branchesCount", branches_count)
    data_frame.set_value(index, "tagsCount", tags_count)
    data_frame.set_value(index, "releasesCount", releases_count)
    return data_frame


def fix_closed_issues(data_frame, index):
    """The sum of merged and closed pull requests must be subtracted from the
    number of closed issues"""
    old_closed_issues_count = data_frame.loc[index, "closed_issues"]
    closed_pull_requests = data_frame.loc[index, "closed_pull_requests"]
    merged_pull_requests = data_frame.loc[index, "merged_pull_requests"]
    data_frame.set_value(index, "closed_issues", old_closed_issues_count -
                         (closed_pull_requests + merged_pull_requests))
    return data_frame
