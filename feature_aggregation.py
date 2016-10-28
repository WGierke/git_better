"""
Aggregate the provided Data Frame with features we receive from the Github REST and GraphQL API
TODO:       commits_count
"""
from tqdm import tqdm
import re
from utils import request_graph_features, website_exists
import json
from utils import get_client, get_last_repos_pagination_page


def aggregate_features(data_frame):
    print "Loading features for " + str(len(data_frame)) + " repositories"
    for index, row in tqdm(data_frame.iterrows(), total=len(data_frame)):
        repo = get_client().get_repo(row['repository'])
        owner = data_frame['owner'][index]
        name = data_frame['name'][index]
        data_frame = add_graph_features(data_frame, index, owner, name)
        data_frame = add_rest_features(data_frame, index, repo)
        data_frame = add_custom_features(data_frame, index, owner, name)
        data_frame = fix_closed_issues(data_frame, index)
    return data_frame


def add_graph_features(data_frame, index, owner, name):
    """closed_issues, open_issues, closed_pull_requests, merged_pull_requests, open_pull_requests,
    projects, watchers, stargazers, forks, mentionableUsers, description"""
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
            if data[k]:
                features[k] = data[k]
            else:
                features[k] = ''
    return features


def add_rest_features(data_frame, index, repo):
    """size, language_rate, readme"""
    data_frame.set_value(index, 'size', repo.size)
    try:
        regex = re.compile('[^a-zA-Z0-9 :\/-]')
        readme = regex.sub('', repo.get_readme().decoded_content)
        data_frame.set_value(index, 'readme', readme)
    except:
        data_frame.set_value(index, 'readme', '')
    sum_loc = sum(repo.get_languages().values())
    languages = repo.get_languages()
    for key in languages.keys():
        data_frame.set_value(index, "LANGUAGE_" + key,
                             languages[key] / float(sum_loc))
    return data_frame


def add_custom_features(data_frame, index, owner, name):
    """isOwnerHomepage, hasHomepage, commitsCount, branchesCount"""
    is_owner_homepage = name.lower() == "{}.github.io".format(owner.lower())
    has_homepage = website_exists("http://{}.github.io/{}".format(owner, name))
    commits_count = get_last_repos_pagination_page("{}/{}/commits?per_page=1".format(owner, name))
    branches_count = get_last_repos_pagination_page("{}/{}/branches?per_page=1".format(owner, name))
    tags_count = get_last_repos_pagination_page("{}/{}/tags?per_page=1".format(owner, name))
    releases_count = get_last_repos_pagination_page("{}/{}/releases?per_page=1".format(owner, name))

    data_frame.set_value(index, "isOwnerHomepage", is_owner_homepage)
    data_frame.set_value(index, "hasHomepage", has_homepage)
    data_frame.set_value(index, "commitsCount", commits_count)
    data_frame.set_value(index, "branchesCount", branches_count)
    data_frame.set_value(index, "tagsCount", tags_count)
    data_frame.set_value(index, "releasesCount", releases_count)
    return data_frame


def fix_closed_issues(data_frame, index):
    """ The sum of merged and closed pull requests must be subtracted from the
    number of closed issues"""
    old_closed_issues_count = data_frame.loc[index, "closed_issues"]
    closed_pull_requests = data_frame.loc[index, "closed_pull_requests"]
    merged_pull_requests = data_frame.loc[index, "merged_pull_requests"]
    data_frame.set_value(index, "closed_issues", old_closed_issues_count -
                         (closed_pull_requests + merged_pull_requests))
    return data_frame
