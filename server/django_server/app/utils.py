import re
from github import Github
from app.training import load_pickle, JOBLIB_DESCRIPTION_PIPELINE_NAME, JOBLIB_README_PIPELINE_NAME


def add_repo_header(repo_list):
    return '''  <div class="repositories">
                    <ul class="list-group">
                        <li class="list-group-item">
                            <div class="row">
                                <div class="col-md-3">
                                    <strong>Your repositories</strong>
                                    <span class="badge pull-right">{}</span>
                                </div>
                                <div class="col-md-3">
                                    <strong>Description</strong>
                                </div>
                                <div class="col-md-2">
                                    <strong>Class (Description)</strong>
                                </div>
                                <div class="col-md-2">
                                    <strong>Class (Readme)</strong>
                                </div>
                                <div class="col-md-2">
                                    <strong>Language</strong>
                                </div>
                            </div>
                        </li>
                        {}
                    </ul>
                </div>'''.format(len(repo_list), '\n'.join(repo_list))


def repo_list_html(repo_infos, descr_model, readme_model):
    repos_html = []
    for i in range(len(repo_infos)):
        name = repo_infos[i][0]
        descr = repo_infos[i][1]
        readme = repo_infos[i][2]
        language = repo_infos[i][3]
        stars = repo_infos[i][4]
        forks = repo_infos[i][5]
        classes = ['DATA', 'DEV', 'DOCS', 'EDU', 'HW', 'WEB']
        predicted_descr = descr_model.predict([descr])[0]
        predicted_readme = readme_model.predict([readme])[0]
        predicted_descr_probas = dict(zip(classes, descr_model.predict_proba([descr])[0]))
        predicted_descr_proba = predicted_descr_probas[predicted_descr]
        predicted_readme_probas = dict(zip(classes, readme_model.predict_proba([readme])[0]))
        predicted_readme_proba = predicted_readme_probas[predicted_readme]

        repos_html.append(
            ''' <li class="list-group-item">
                    <div class="row">
                        <div class="col-md-3">
                            <a href="https://github.com/{name}">{name}</a>
                        </div>
                        <div class="col-md-3">
                            {description}
                        </div>
                        <div class="col-md-2">
                            {pred_descr} ({pred_descr_proba:0.4f})
                        </div>
                        <div class="col-md-2">
                            {pred_readme} ({pred_readme_proba:0.4f})
                        </div>
                        <div class="col-md-2">
                            {language}
                        </div>
                    </div>
                </li>
                    '''.format(name=name, description=descr,
                    pred_descr=predicted_descr, pred_descr_proba=predicted_descr_proba,
                    pred_readme=predicted_readme, pred_readme_proba=predicted_readme_proba,
                    language=language, stars=stars, forks=forks))
    return add_repo_header(repos_html)


def build_repo_html(token):
    repo_html = 'No valid GitHub API token received'
    if token != '':
        descr_model = load_pickle(JOBLIB_DESCRIPTION_PIPELINE_NAME)
        readme_model = load_pickle(JOBLIB_README_PIPELINE_NAME)
        repo_infos = []
        print "Token: " + token
        client = Github(token)
        user = client.get_user()
        readme_regex = re.compile('[^a-zA-Z0-9 :\/-]')
        for repo in user.get_repos():
            readme = ''
            try:
                readme = repo.get_readme().decoded_content
            except:
                pass
            print readme[:10]
            info = (repo.full_name, repo.description or '',
                    readme_regex.sub('', readme),
                    repo.language, repo.stargazers_count, repo.forks_count)
            repo_infos.append(info)
        return repo_list_html(repo_infos, descr_model, readme_model)
    return repo_html
