import requests
import os
import sys
import time
import ConfigParser
import json
import logging
from github import Github

GITHUB_CLIENT = "Client"
# List of username,token,remainingGraphQlRequests touples
USERNAME_TOKENS = []
TOKEN_POINTER = 0


def load_config():
    if os.path.isfile("config.ini"):
        global USERNAME_TOKENS
        global GITHUB_CLIENT
        config = ConfigParser.RawConfigParser(allow_no_value=False)
        config.read('config.ini')
        USERNAME_TOKENS = get_username_tokens(config)
        print "Loaded " + str(len(USERNAME_TOKENS)) + " user tokens"
        GITHUB_CLIENT = Github(login_or_token=get_token())
        ensure_valid_token()
    else:
        print "Please provide a config.ini file which includes your Github credentials (see example.config.ini for an example)"
        sys.exit(1)

def get_username_tokens(config):
    index = 1
    tokens = []
    while True:
        try:
            username = config.get("github" + str(index), "username")
            token = config.get("github" + str(index), "token")
            remaining = get_remaining_graphql_requests(username, token)
            tokens.append((username, token, remaining))
            index += 1
        except ConfigParser.NoSectionError, e:
            return tokens
        except Exception, e:
            print e
            return tokens


def get_username():
    global USERNAME_TOKENS
    global TOKEN_POINTER
    return USERNAME_TOKENS[TOKEN_POINTER][0]


def get_token():
    global USERNAME_TOKENS
    global TOKEN_POINTER
    return USERNAME_TOKENS[TOKEN_POINTER][1]


def get_remaining():
    global USERNAME_TOKENS
    global TOKEN_POINTER
    return USERNAME_TOKENS[TOKEN_POINTER][2]


def decrease_remaining():
    global USERNAME_TOKENS
    global TOKEN_POINTER
    USERNAME_TOKENS[TOKEN_POINTER] = (USERNAME_TOKENS[TOKEN_POINTER][0], USERNAME_TOKENS[TOKEN_POINTER][1], USERNAME_TOKENS[TOKEN_POINTER][2]-1)

def get_client():
    global GITHUB_CLIENT
    return GITHUB_CLIENT


def get_remaining_graphql_requests(username, token):
    r = get_rate_limit(username, token)
    response = json.loads(r.content)
    return int(response["resources"]["graphql"]["remaining"])


def get_rate_limit(username, token):
    return requests.get('https://api.github.com/rate_limit', auth=(username, token))


def ensure_valid_token():
    global USERNAME_TOKENS
    global TOKEN_POINTER
    if get_remaining() < 1:
        print "Remaining requests for " + get_username() + " is 0"
        while True:
            for index, user_key in enumerate(USERNAME_TOKENS):
                if user_key[2] > 0:
                    TOKEN_POINTER = index
                    print "Chose token for " + get_username()
                    return
            print "No remaining requests for any token available. Checking again in 5 minutes."
            time.sleep(300)
            load_config()


def request_graph_api(query):
    ensure_valid_token()
    query = query.replace("'", "\"")
    headers = {'Authorization': 'bearer ' + get_token()}
    response = requests.post('https://api.github.com/graphql', headers=headers, data=json.dumps({"query": query}))
    r = json.loads(response.content)
    if "message" in r.keys() and "API rate limit exceeded for" in r["message"]:
        ensure_valid_token()
        response = requests.post('https://api.github.com/graphql', headers=headers, data=json.dumps({"query": query}))
    else:
        decrease_remaining()
    return response.content


def request_graph_features(repo_owner, repo_name):
    query = """{{
      repositoryOwner(login: "{0:s}") {{
        repository(name: "{1:s}") {{
          open_pull_requests: pullRequests(states: [OPEN]) {{
            totalCount
          }}
          merged_pull_requests: pullRequests(states: [MERGED]) {{
            totalCount
          }}
          closed_pull_requests: pullRequests(states: [CLOSED]) {{
            totalCount
          }}
          open_issues: issues(states: [OPEN]) {{
            totalCount
          }}
          closed_issues: issues(states: [CLOSED]) {{
            totalCount
          }}
          projects {{
            totalCount
          }}
          watchers {{
            totalCount
          }}
          stargazers {{
            totalCount
          }}
          forks {{
            totalCount
          }}
          description
          mentionableUsers {{
            totalCount
          }}
          mentionableUsers {{
            totalCount
          }}
        }}
      }}
    }}""".format(repo_owner, repo_name)
    return request_graph_api(query)


def get_response(url):
    headers = {'Authorization': 'token %s' % get_token()}
    try:
        return requests.get(url, headers=headers)
    except Exception, e:
        logging.error(e)
        return None


def website_exists(url, prefix='', only_headers=False):
    try:
        if only_headers:
            response = requests.head(prefix + url)
        else:
            response = get_response(prefix + url)
        return response.status_code < 400
    except:
        return False


def get_last_pagination_page(url):
    try:
        response = get_response(url)
        if 'Link' in response.headers:
            return int(response.headers['Link'].split(',')[1].split("&page=")[1].split(">")[0])
        else:
            return len(json.loads(response.content))
    except:
        return 0


def get_last_repos_pagination_page(url):
    return get_last_pagination_page("https://api.github.com/repos/" + url)
