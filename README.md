# git_better [![CircleCI](https://circleci.com/gh/WGierke/git_better.svg?style=svg&circle-token=3fb4bac4bb656bc2e7b9dbb6d9dc77a303bd240c)](https://circleci.com/gh/WGierke/git_better)
[Demo](http://git-better.herokuapp.com/)
## Installation
### Repository Classification
- Make sure that you've installed Python 2.7
- Install [xgboost](http://xgboost.readthedocs.io/en/latest/build.html) manually
- Create a virtual environment and install all dependencies  
```bash
virtualenv -p /usr/bin/python2.7 venv
source venv/bin/activate`  
pip install -r requirements.txt
```
- Download the NLTK corpus
```bash
python -m nltk.downloader all
```
- Create a [personal access token](https://github.com/settings/tokens), grant "Full control of private repositories" (repo) and put it in your config.ini file  
```bash
cp example.config.ini config.ini
```
- Accept the [Github Pre-release Program agreement](https://github.com/prerelease/agreement) so your access token can also use GitHub's GraphQL API  

### Django Server
#### Manual
- Psycopg2 needs postgresql-devel
```bash
apt-get install -y libpq-dev
```
- Sklearn needs Tkinter
```bash
apt-get install python-tk
```
- Migrate the database  
```bash
python server/manage.py migrate
```
- Run the server (per default on port 8000)  
```bash
python server/start_server.py
```
- You can access the application now on [localhost:8000](http://localhost:8000)  

#### Docker
- Install [Docker](https://docs.docker.com/)
- Build the docker image
```bash
docker build -t git_better .
```
- Create a container from the image and run it in the background
```bash
docker run -d -p 8000:8000 git_better
```
- You can access the application now on [localhost:8000](http://localhost:8000)  

## Usage
To predict repository labels based on your own training data or based on pre-trained models, follow the instructions of our main script:   
`python app/main.py --help`  
As an example, to classify the input data from the challenge repository using our pre-trained models, run  
`python app/main.py -i data/example-input.txt`

To visualize the data with the TensorBoard Embedding Projector, run `python app/embedding_visualization.py` and start tensorboard with `tensorboard --logdir log/`. Tensorboard will display the port on which the server listens, open `localhost:[port]` with your browser (standard port is 6006).

## Testing
To test whether the app works correctly, simply run `python -m unittest discover`

## Deployment on Heroku
- Install the [Heroku Container Registry and Runtime](https://devcenter.heroku.com/articles/container-registry-and-runtime) and log in   
- Build the Docker image, tag it, push it to Heroku and open the website  
```
docker build -t git_better .  
docker tag git_better registry.heroku.com/git-better/web  
docker push registry.heroku.com/git-better/web  
heroku open  
```
