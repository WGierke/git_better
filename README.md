# git_better [![CircleCI](https://circleci.com/gh/WGierke/git_better.svg?style=svg&circle-token=3fb4bac4bb656bc2e7b9dbb6d9dc77a303bd240c)](https://circleci.com/gh/WGierke/git_better)
[Demo](git-better.herokuapp.com/)
## Installation
### Repository Classification
- Make sure that you've installed Python 2.7
- [Install xgboost](http://xgboost.readthedocs.io/en/latest/build.html)
- Create a virtual environment and install all dependencies  
`virtualenv -p /usr/bin/python2.7 venv && source venv/bin/activate`  
`pip install -r requirements.txt`  
- Create a [personal access token](https://github.com/settings/tokens) and put it in your config.ini file  
`cp example.config.ini config.ini`  
- Accept the [Github Pre-release Program agreement](https://github.com/prerelease/agreement) so your access token can also use GitHub's GraphQL API

### Django Server
- Sync the database  
`python manage.py syncdb`  
- Run the server  
`python manage.py runserver`  

## Usage
To fetch all features for the repositories specified in the CSV located at `TRAINING_DATA_PATH` and save them in `data/processed_data.csv` run: `python app/process.py`

##Testing
To test whether the app works correctly, simply run `python -m unittest discover`
