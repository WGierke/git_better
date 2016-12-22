from gittle import Gittle
import os
from binaryornot.check import is_binary

REPOS_DIR = "repo/"

def clone_repo(repository_name, clone_url):
    repo_dir = os.path.join(REPOS_DIR,repository_name)

    # check if repository is already cloned
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)

        repo = Gittle.clone(clone_url, repo_dir)
    return repo_dir

def merge_files(repo_dir):
    # check if source code files are already merged
    if not os.path.exists(os.path.join(repo_dir,"merged_source.txt")):
        with open(os.path.join(repo_dir,"merged_source.txt"), "w") as outfile:
            for subdir, dirs, files in os.walk(repo_dir):
                for file in files:
                    filepath = os.path.join(subdir, file)

                    # is_binary is sth. like a heuristic
                    if(not is_binary(filepath)):
                        with open(filepath, "rb") as infile:
                            outfile.write(infile.read())


if __name__ == '__main__':
    repo_dir = clone_repo('gittle', 'https://github.com/FriendCode/gittle.git')
    merge_files(repo_dir)