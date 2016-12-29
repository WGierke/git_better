from app.evaluation import get_cleaned_processed_df
from app.training import find_best_text_pipeline
from gittle import Gittle
import os
from binaryornot.check import is_binary
import pandas as pd
import traceback

REPOS_DIR = "repo/"

def clone_repo(owner, name, clone_url, pull=True):
    owner_dir = os.path.join(REPOS_DIR,owner)
    repo_dir = os.path.join(owner_dir,name)
    print owner
    print name
    print clone_url

    # check if repository is already cloned
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
        print 'Cloning repository ' + owner + '/' + name + '...'
        try:
            repo = Gittle.clone(clone_url, repo_dir)
        except KeyError, e:
            print 'KeyError while cloning... '
            traceback.print_exc()
            pass
        except Exception:
            traceback.print_exc()
            pass

    elif pull==True:
        repo = Gittle(repo_dir, origin_uri=clone_url)
        print 'Pulling repository ' + owner + '/' + name + '...'
        try:
            repo.pull()
        except KeyError, e:
            print 'KeyError while pulling... '
            traceback.print_exc()
            pass
        except Exception:
            traceback.print_exc()
            pass
    return repo_dir

def merge_commit_messages(repo_dir, override=False):
    # check if commit messages are already merged
    if override or not os.path.exists(os.path.join(repo_dir,"merged_commit_messages.txt")):
        with open(os.path.join(repo_dir,"merged_commit_messages.txt"), "w") as outfile:
            repo = Gittle(repo_dir)
            commits = repo.commit_info()
            for commit in commits:
                commit_message = commit.get('message')
                # add \n if necessary
                if os.linesep not in commit_message[-4:]:
                    commit_message = commit_message + os.linesep
                outfile.write(commit_message)
    return open(os.path.join(repo_dir,"merged_commit_messages.txt"), "r").read()


def merge_files(repo_dir, override=False):
    # check if source code files are already merged
    if override or not os.path.exists(os.path.join(repo_dir,"merged_source.txt")):
        with open(os.path.join(repo_dir,"merged_source.txt"), "w") as outfile:
            for subdir, dirs, files in os.walk(repo_dir):
                for file in files:
                    filepath = os.path.join(subdir, file)

                    # is_binary() is sth. like a heuristic
                    if(not is_binary(filepath)):
                        with open(filepath, "rb") as infile:
                            outfile.write(infile.read())
    return open(os.path.join(repo_dir,"merged_source.txt"), "r").read()

# TODO: Is the folder name also interesting?
def merge_file_names(repo_dir, override=False):
    # check if file names are already merged
    if override or not os.path.exists(os.path.join(repo_dir,"merged_file_names.txt")):
        with open(os.path.join(repo_dir,"merged_file_names.txt"), "w") as outfile:
            for subdir, dirs, files in os.walk(repo_dir):
                for file in files:
                    # exclude git specific and repository mining files, they don't add extra knowledge
                    if ".git" not in subdir \
                            and "merged_commit_messages.txt" not in file \
                            and "merged_file_names.txt" not in file \
                            and "merged_source.txt" not in file:
                        outfile.write(file+os.linesep)
    return open(os.path.join(repo_dir,"merged_file_names.txt"), "r").read()


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
        merge_commit_messages(repo_dir)
        merge_file_names(repo_dir)
        df_merged_text = df_merged_text.append(
            pd.DataFrame([[full_name,merged_text,label]],columns=['full_name','source code','label']))

    find_best_text_pipeline(df_merged_text['source code'], df_merged_text['label'])