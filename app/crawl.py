import re
import time
import requests
import json
import os
from lxml import html
from utils import load_config, get_username, get_token

REPO_REGEX = r"(?P<user>\w+)(?=\/(?P=user)\.github\.io)"
test_str = "WGierke/wgierke.github.io"
REPO_LINK_FILE = "data/{label}_links.txt"


def save_github_search_repos(q='github.io', label='web'):
    load_config()
    page = 0
    url = "https://api.github.com/search/repositories?q=" + q + " + &page={page}"
    links = get_label_links(label=label)

    try:
        r = None
        while page == 0 or r.status_code == 200:
            page += 1
            print page
            r = requests.get(url.format(page=page), auth=(get_username(), get_token()))
            repos = json.loads(r.text)["items"]
            for repo in repos:
                name = repo["full_name"]
                links_length = len(links)
                links.add("https://github.com/" + name)
                if links_length != len(links):
                    print "Added: " + name
            time.sleep(2)
        write_label_links(links, label)
    except:
        write_label_links(links, label)
        print links


def save_showcases_repos(showcase_name='web-application-frameworks', label='dev'):
    page = requests.get('https://github.com/showcases/' + showcase_name)
    tree = html.fromstring(page.content)
    repos = tree.xpath("/html/body/div[4]/div[2]/div[3]/div/div[1]/ul/li")
    links = get_label_links(label=label)
    old_length = len(links)

    for repo in repos:
        owner = repo.xpath("h3/a/span/text()")[0].replace(" / ", "")
        name = repo.xpath("h3/a/text()")[1].replace("\n", "")
        links.add("https://github.com/{}/{}".format(owner, name))
    print "Added {} links".format(len(links) - old_length)
    write_label_links(links, label)


def get_label_links(label='web'):
    web_link_file = REPO_LINK_FILE.format(label=label)
    content = None

    if os.path.isfile(web_link_file):
        with open(web_link_file, "r") as f:
            content = f.read()

    if content:
        links = set(json.loads(content))
    else:
        links = set()
    return links


def write_label_links(links, label='web'):
    links = list(links)
    web_link_file = REPO_LINK_FILE.format(label=label)
    overwrite_file_with_content(json.dumps(links, indent=4), web_link_file)
    return


def overwrite_file_with_content(content, file_path):
    with open(file_path, 'w'): pass
    with open(file_path, "w") as f:
        f.write(content)
