import re
import time
import requests
import json
import os
from utils import load_config, get_username, get_token

REPO_REGEX = r"(?P<user>\w+)(?=\/(?P=user)\.github\.io)"
test_str = "WGierke/wgierke.github.io"
REPO_LINK_FILE = "data/{label}_links.txt"


def get_repo_links(label='web'):
    load_config()
    page = 0
    url = "https://api.github.com/search/repositories?q=github.io&page={page}"
    links = None
    content = None
    web_link_file = REPO_LINK_FILE.format(label=label)

    if os.path.isfile(web_link_file):
        with open(web_link_file, "r") as f:
            content = f.read()

    if content:
        links = set(json.loads(content))
    else:
        links = set()

    try:
        r = None
        while page == 0 or r.status_code == 200:
            page += 1
            print page
            r = requests.get(url.format(page=page), auth=(get_username(), get_token()))
            repos = json.loads(r.text)["items"]
            for repo in repos:
                name = repo["full_name"]
                matches = re.finditer(REPO_REGEX, name, re.IGNORECASE)
                for _, match in enumerate(matches):
                    links_length = len(links)
                    links.add("https://github.com/" + name)
                    if links_length != len(links):
                        print "Added: " + name
            time.sleep(2)
        overwrite_file_with_content(json.dumps(links, indent=4), web_link_file)
    except:
        links = list(links)
        overwrite_file_with_content(json.dumps(links, indent=4), web_link_file)
        print links


def overwrite_file_with_content(content, file_path):
    with open(file_path, 'w'): pass
    with open(file_path, "w") as f:
        f.write(content)
