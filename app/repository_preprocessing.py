from app.evaluation import get_cleaned_processed_df
from app.training import find_best_text_pipeline
from gittle import Gittle
import os
from binaryornot.check import is_binary
import pandas as pd
import traceback
import codecs
from sklearn.externals import joblib

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
        with codecs.open(os.path.join(repo_dir,"merged_commit_messages.txt"), "w", "utf-8") as outfile:
            repo = Gittle(repo_dir)
            commits = repo.commit_info()
            for commit in commits:
                commit_message = commit.get('message')
                try:
                    commit_message.decode('utf-8')
                    # add \n if necessary
                    if os.linesep not in commit_message[-4:]:
                        commit_message = commit_message + os.linesep
                    outfile.write(commit_message)
                except UnicodeError:
                    pass
    return open(os.path.join(repo_dir,"merged_commit_messages.txt"), "r").read()


def merge_files(repo_dir, file_name, override=False):
    # check if source code files are already merged
    if override or not os.path.exists(os.path.join(repo_dir,file_name)):
        with codecs.open(os.path.join(repo_dir,file_name), "w", "utf-8") as outfile:
            for subdir, dirs, files in os.walk(repo_dir):
                for file in files:
                    filepath = os.path.join(subdir, file)

                    # is_binary() is sth. like a heuristic
                    try:
                        if(not is_binary(filepath)) \
                                and ".git" not in subdir \
                                and "merged_commit_messages.txt" not in file \
                                and "merged_file_names.txt" not in file \
                                and "merged_wiki.txt" not in file \
                                and "merged_source.txt" not in file:
                            with open(filepath, "rb") as infile:
                                try:
                                    f = codecs.open(filepath, encoding='utf-8', errors='strict')
                                    for line in f:
                                        pass
                                    outfile.write(infile.read())
                                except UnicodeDecodeError:
                                    print "invalid utf-8"
                                except Exception, e:
                                    print "Something went wrong while reading file" + str(e)
                    except IOError, e:
                        print "File does not exist" + str(e)
                        pass
    return open(os.path.join(repo_dir,file_name), "r").read()

# TODO: Is the folder name also interesting?
def merge_file_names(repo_dir, override=False):
    # check if file names are already merged
    if override or not os.path.exists(os.path.join(repo_dir,"merged_file_names.txt")):
        with codecs.open(os.path.join(repo_dir,"merged_file_names.txt"), "w", "utf-8") as outfile:
            for subdir, dirs, files in os.walk(repo_dir):
                for file in files:
                    # exclude git specific and repository mining files, they don't add extra knowledge
                    if ".git" not in subdir \
                            and "merged_commit_messages.txt" not in file \
                            and "merged_file_names.txt" not in file \
                            and "merged_wiki.txt" not in file \
                            and "merged_source.txt" not in file:
                        try:
                            file.decode('utf-8')
                            outfile.write(file+os.linesep)
                        except UnicodeError:
                            pass
                    try:
                        dirname = dirs[-1]
                        dirname.decode('utf-8')
                        outfile.write(dirname+os.linesep)
                    except UnicodeError:
                        pass
    return open(os.path.join(repo_dir,"merged_file_names.txt"), "r").read()


# Data necessary for building the data frame with source code
def get_repo_clone_data(data_frame):
    #data_frame = pd.DataFrame([["octocat/Hello-World","https://github.com/octocat/Hello-World.git","DEV"],
    #                           ["FriendCode/gittle", "https://github.com/FriendCode/gittle.git","DEV"]],
    #                          columns=["full_name","clone_url","label"])
    labels = data_frame.pop("label")

    owners = data_frame.pop("owner")
    names = data_frame.pop("name")

    cloneUrls = []
    for owner, name in zip(owners, names):
        cloneUrls.append('https://github.com/' + owner + '/' + name + '.git')

    return cloneUrls, labels, owners, names

# available options for mode are 'source_code', 'commit_messages' and 'file_names' def
def get_data_repos(cloneUrls, labels, owners, names, mode='source_code', pull=False, override=True):
    df_merged_text = pd.DataFrame(columns=[['full_name','source code','label']])

    #i = 0
    for cloneUrl, label, owner, name in zip(cloneUrls, labels, owners, names):
        # if owner not in ['DataScienceSpecialization', 'cdcepi', 'gygy', 'koolshare', 'Gaohaoyang', 'GoogleWebComponents']:
            # if i%10==0:
        repo_dir = clone_repo(owner, name, cloneUrl)
        if mode=='source_code':
            merged_text = merge_files(repo_dir, file_name='merged_source.txt', override)
        elif mode=='commit_messages':
            merged_text = merge_commit_messages(repo_dir)
        elif mode=='file_names':
            merged_text = merge_file_names(repo_dir)
        elif mode=='wiki':
            repo_dir = repo_dir[:-4]
            repo_dir = repo_dir + '.wiki.git'
            merged_text = merge_files(repo_dir, file_name='merged_wiki.txt', override)
        else:
            print('Not supported mode')
        df_merged_text = df_merged_text.append(
            pd.DataFrame([[owner, name, merged_text,label]],columns=['owner', 'name','source code','label']))
            # i+=1
            # if i==1000:
            #     break
    return df_merged_text

if __name__ == '__main__':
    if not os.path.exists(os.path.join(REPOS_DIR,'merged_text.pickle')):
        cloneUrls, labels, owners, names = get_repo_clone_data(get_cleaned_processed_df())
        df_merged_text = get_data_repos(cloneUrls, labels, owners, names)
        df_merged_text.to_pickle(os.path.join(REPOS_DIR,'merged_text.pickle'))
    else:
        df_merged_text = pd.read_pickle(os.path.join(REPOS_DIR,'merged_text.pickle'))

    pipeline = find_best_text_pipeline(df_merged_text['source code'], df_merged_text['label'])

    joblib.dump(pipeline, 'source-code-pipeline.pkl')