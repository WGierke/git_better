from app.evaluation import get_cleaned_processed_df
from app.training import find_best_text_pipeline
from gittle import Gittle
import os
from binaryornot.check import is_binary
import pandas as pd

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
    return open(os.path.join(repo_dir,"merged_source.txt"), "r").read()


if __name__ == '__main__':
    #data_frame = get_cleaned_processed_df()
    data_frame = pd.DataFrame([["octocat/Hello-World","https://github.com/octocat/Hello-World.git","DEV"],
                               ["FriendCode/gittle", "https://github.com/FriendCode/gittle.git","DEV"]],
                              columns=["full_name","clone_url","label"])
    cloneUrls = data_frame.pop("clone_url")
    labels = data_frame.pop("label")
    # TODO: use repo name and owner.login instead of full_name, '/' in full_name only works on UNIX
    full_names = data_frame.pop("full_name")

    df_merged_text = pd.DataFrame(columns=[['full_name','source code','label']])

    for cloneUrl, label, full_name in zip(cloneUrls, labels, full_names):
        repo_dir = clone_repo(full_name, cloneUrl)
        merged_text = merge_files(repo_dir)
        df_merged_text = df_merged_text.append(
            pd.DataFrame([[full_name,merged_text,label]],columns=['full_name','source code','label']))

    find_best_text_pipeline(df_merged_text['source code'], df_merged_text['label'])