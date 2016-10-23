from tqdm import tqdm


def add_count_features(data_frame, repo, index):
    data_frame.set_value(index, 'watchers_count', repo.watchers_count)
    data_frame.set_value(index, 'forks_count', repo.forks_count)
    data_frame.set_value(index, 'open_issues_count', repo.open_issues_count)
    data_frame.set_value(index, 'stargazers_count', repo.stargazers_count)
    return data_frame


def add_text_features(data_frame, repo, index):
    data_frame.set_value(index, 'description', repo.description)
    data_frame.set_value(index, 'readme', repo.get_readme().decoded_content)
    return data_frame


def add_relative_language(data_frame, repo, index):
    sum_loc = sum(repo.get_languages().values())
    languages = repo.get_languages()
    for key in languages.keys():
        data_frame.set_value(index, "LANGUAGE_" + key, languages[key] / float(sum_loc))
    return data_frame


def aggregate_features(github_client, data_frame):
    print "Loading features for " + str(len(data_frame)) + " repositories"
    for index, row in tqdm(data_frame.iterrows(), total=len(data_frame)):
        repo = github_client.get_repo(row['repository'])
        data_frame = add_count_features(data_frame, repo, index)
        data_frame = add_text_features(data_frame, repo, index)
        data_frame = add_relative_language(data_frame, repo, index)
    return data_frame
