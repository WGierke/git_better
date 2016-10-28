import requests
import os
import sys
import ConfigParser
import json
from github import Github

PERSONAL_ACCESS_TOKEN = "Token"
GITHUB_CLIENT = "Client"


def load_config():
    if os.path.isfile("config.ini"):
        global GITHUB_CLIENT
        global PERSONAL_ACCESS_TOKEN
        config = ConfigParser.RawConfigParser(allow_no_value=False)
        config.read('config.ini')
        username = config.get("github", "username")
        password = config.get("github", "password")
        GITHUB_CLIENT = Github(username, password)
        PERSONAL_ACCESS_TOKEN = config.get("github", "token")
    else:
        print "Please provide a config.ini file which includes your Github credentials"
        sys.exit(1)


def get_client():
    return GITHUB_CLIENT


def get_token():
    return PERSONAL_ACCESS_TOKEN


def request_graph_api(query):
    query = query.replace("'", "\"")
    headers = {'Authorization': 'bearer ' + PERSONAL_ACCESS_TOKEN}
    return requests.post('https://api.github.com/graphql', headers=headers, data=json.dumps({"query": query})).content


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
    return requests.get(url)


def website_exists(url):
    return get_response(url).status_code < 400


def get_last_pagination_page(url):
  try:
    link_header = get_response(url).headers['Link']
    return int(link_header.split(',')[1].split("&page=")[1].split(">")[0])
  except:
    return 1
