from github import Github
from app.training import load_pickle, JOBLIB_DESCRIPTION_PIPELINE_NAME

def add_repo_header(repo_list):
    return '''  <div class="repositories">
                    <ul class="list-group">
                        <li class="list-group-item">
                            <div class="row">
                                <div class="col-md-3">
                                    <strong>Your repositories</strong>
                                    <span class="badge pull-right">{}</span>
                                </div>
                                <div class="col-md-4">
                                    <strong>Description</strong>
                                </div>
                                <div class="col-md-1">
                                    <strong>Class (Description)</strong>
                                </div>
                                <div class="col-md-2">
                                    <strong>Language</strong>
                                </div>
                                <div class="col-md-1">
                                    <strong>Stars</strong>
                                </div>
                                <div class="col-md-1">
                                    <strong>Forks</strong>
                                </div>
                            </div>
                        </li>
                        {}
                    </ul>
                </div>'''.format(len(repo_list), '\n'.join(repo_list))


def repo_list_html(repo_infos, model):
    repos_html = []
    for i in range(len(repo_infos)):
        name = repo_infos[i][0]
        descr = repo_infos[i][1]
        language = repo_infos[i][2]
        stars = repo_infos[i][3]
        forks = repo_infos[i][4]
        classified = model.predict([descr])[0]
        classes = ['DATA', 'DEV', 'DOCS', 'EDU', 'HW', 'WEB']
        classified_probas = dict(zip(classes, model.predict_proba([descr])[0]))
        classified_proba = classified_probas[classified]

        repos_html.append(
            ''' <li class="list-group-item">
                    <div class="row">
                        <div class="col-md-3">
                            <a href="https://github.com/{name}">{name}</a>
                        </div>
                        <div class="col-md-4">
                            {description}
                        </div>
                        <div class="col-md-1">
                            {classified} ({classified_proba:0.4f})
                        </div>
                        <div class="col-md-2">
                            {language}
                        </div>
                        <div class="col-md-1">
                            <span class="star-badge pull-left">
                                {stars}
                                <span class="octicon octicon-star"></span>
                            </span>
                        </div>
                        <div class="col-md-1">
                            <span class="star-badge pull-left">
                                {forks}
                                <span class="octicon octicon-repo-forked"></span>
                            </span>
                        </div>
                    </div>
                </li>
                    '''.format(name=name, description=descr, classified=classified, classified_proba=classified_proba, language=language, stars=stars, forks=forks))
    return add_repo_header(repos_html)


def build_repo_html(token):
    repo_html = 'No valid GitHub API token received'
    if token != '':
        model = load_pickle(JOBLIB_DESCRIPTION_PIPELINE_NAME)
        repo_infos = []
        print "Token: " + token
        client = Github(token)
        user = client.get_user()
        for repo in user.get_repos():
            info = (repo.full_name, repo.description or '', repo.language,
                    repo.stargazers_count, repo.forks_count)
            repo_infos.append(info)
        return repo_list_html(repo_infos, model)
    return repo_html
