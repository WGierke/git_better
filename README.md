# git_better [![CircleCI](https://circleci.com/gh/WGierke/git_better.svg?style=svg&circle-token=3fb4bac4bb656bc2e7b9dbb6d9dc77a303bd240c)](https://circleci.com/gh/WGierke/git_better)

## Installation
- Make sure that you've installed Python 2.7
- Create a virtual environment and install all dependencies  
`virtualenv -p /usr/bin/python2.7 venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  
- Create a [personal access token](https://github.com/settings/tokens) and put it in your config.ini file  
`cp example.config.ini config.ini`  
- Accept the [Github Pre-release Program agreement](https://github.com/prerelease/agreement) so your access token can also use GitHub's GraphQL API

## Usage
To fetch all features for the repositories specified in the CSV located at `TRAINING_DATA_PATH` and save them in `data/processed_data.csv` run: `python process.py`

##Testing
To test whether the app works correctly, simply run `python -m unittest discover`