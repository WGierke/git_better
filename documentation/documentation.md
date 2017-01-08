# InformatiCup2017 Documentation
## 1. Challenge Description
This years InformatiCup challenge was to classify GitHub repositories automatically based on given class descriptions and sample data.
In this work we present how we explored the given data, detected relevant features and built an application that predicts repository labels using different machine learning algorithms.


## 2. Data Exploration
This section explains how we extended the training data set and how we explored it using different dimension reduction algorithms and visualization tools.

### 2.1 Data Retrieval
The corresponding repository of the challenge includes 30 labeled repositories and 31 repositories that can be used as validation data.
It wouldn't be possible to train convincing prediction models using only these provided data sets.
To extend the amount of available training data (and as a first step to reduce overfitting), we used the GitHub Search API, GitHub Showcases and automated as well as manual Google searches to retrieve more data.
One can find the amount of retrieved, labeled repositories and their origin in the following table.

| Amount | Label         | Origin  |
| ------:|:-------------:| -------|
| 9 | DATA | Manual Google search for Open Data repositories |
| 82  | DATA | Repositories of Github user 'datasets' |
| 17  | EDU | Github Search for "course, material" |
| 17  | DOCS | Github Search for "documentation" |
| 423 | WEB | Google Search for "site:.github.io" |
| 58  | HW | GitHub Search for "homework, assignments, solution" |
| 13  | DEV | Showcases "Virtual Reality" |
| 12  | DEV | Showcases "Software Development Tools" |
| 14  | DEV | Showcases "Front-end JavaScript frameworks" |
| 20  | DEV | Showcases "DevOps tools" |
| 16  | DEV | Showcases "Text editors" |
| 24  | DEV | Showcases "Game Engines" |
| 27  | DEV | Showcases "Web Application Frameworks" |
| 42  | DEV | Showcases "Programming Languages" |
| 180 | DOCS | GitHub Repo Content: Awesome Repos |
| 6 | DATA | Showcases "Open Data" |
| 86  | HW | Github Search for "homework, solution" |

Overall, we were able to collect 1412 labeled repositories.  

**Training Data Distribution**  

![Training Data Distribution](https://cloud.githubusercontent.com/assets/6676439/20865343/ae126750-ba17-11e6-9a85-e55c9fb7224f.png)  

As one can see, we tried to use key words for automated searching that are as close to the words that were used to describe the different classes as possible.
Though, it's still possible that the collected training data is biased as we actively selected repositories by searching for them. As an extension, an approach that could minimize this bias would be to randomly select repositories (e.g. from the GHTorrent project) and label them manually. For the beginning, however, we neither had the time nor the manpower to label a large amount of repositories manually.
Since the difficulty to collect data entries of a certain label differed, we ended up with unbalanced training data.
As the class label distribution affects some classifiers heavily, we trained the models on randomly undersampled training data.

### 2.2 Data Analysis
To get a better idea of how the relationships between the data entries look like in a higher dimensional space, we used principal component analysis (PCA) and t-distributed stochastic neighbor embedding (t-SNE) to reduce the  complexity of the data to 2D while retaining the principal components respectively the distances between the data points.
The following figure visualizes the distribution of the labeled data entries using t-SNE.
![](https://cloud.githubusercontent.com/assets/6676439/21290072/ad44ed02-c4ad-11e6-8314-a078c3b1c853.png)
You can find the complete code to generate the figure in the [t-SNE Visualization Notebook](https://github.com/WGierke/git_better/blob/master/t-SNE%20Visualization.ipynb). To explore the data interactively and in a three dimensional reduction you can use the [tensorflow embedding projector setup](https://github.com/WGierke/git_better#usage). [maybe add footnotes]
One can notice that the "DOCS" repositories build a cluster while it seems to be more complicated to separate the other classes.

We also used t-SNE to visualize the similarity between our retrieved training data and the given validation data.
![](https://cloud.githubusercontent.com/assets/6676439/21290073/ad4817c0-c4ad-11e6-92e2-0d983ed39677.png)

Since the validation data does not form separate clusters or outliers, we could assume that testing the learned models on the validation data is a good way to verify how well the models generalize.
On the other side, the validation data only contains roughly 30 data entries which is not enough to give reliable statements about the model performances.
Furthermore, the fact that the validation data seems to be selected manually implies that it's also biased.
Thus, perfect validation data would be a lot of randomly selected repositories that have been labeled manually.
The [additional data sets](https://github.com/InformatiCup/InformatiCup2017/tree/master/additional_data_sets) from another team could allow us to validate our models better even if they are also biased.
As already mentioned, a perfect training and validation set would only contain repositories that have been sampled randomly and labeled manually.

## 3. Prediction Model
When a model fits its training data too well and doesn't learn to generalize, it's considered to "overfit".
This can occur if the model is too complex so it learns the training data by heart instead of understanding how to solve the given problem in general.
To prevent this, we collected more training data than we already received by the challenge, we applied regularization to hinder the model becoming too complex and we used ensemble learning.  
By collecting more data than already given we created a bigger problem domain that needs to be understood by the model.  
Regularization adds a measure of the models complexity to the cost function that needs to be optimized.
Thus, the model does not only try to solve the given problem but it also tries to do that keeping itself as simple as possible.  
Ensemble learning uses multiple trained models to calculate one final prediction.
These models are trained using distinct features so they're not related to each other.
The assumption is that when one model makes a mistake predicting a label, the other models don't make a mistake in this situation so the (correct predicted) label of the other models is returned.
To decide which model prediction is the correct one we used the Majority Rule algorithm.
One model was trained on the numerical features of a repository, one on the description, one on the content of the readme and one on the source code of each repository.  
The following chapters will explain how we retrieved and cleaned the data for each model, how we selected relevant features  and how we developed the prediction model.

### 3.1 Training and Test Data Set
To train and evaluate the classifiers, we used a train/test/validation split.  
First, the collected training data was splitted in a train and a test split in a stratified manner.
This ensured that the distribution of class labels was balanced in both splits.  
The classifiers were then trained on the train split and their accuracy was evaluated on the test split.
To calculate their final quality, we evaluated them on the validation data.

### 3.2 Classification Using Numeric Metadata of Repositories
To develop classifiers based on numeric metadata of repositories, we used the following features:

|Feature Name|Description|
|------------|-----------|
|watchers | Number of users who watch the repo |
|mentionableUsers | Number of mentionable users (collaborators, contributors, ...) |
|open_pull_requests | Number of open pull requests |
|closed_pull_requests | Number of closed pull requests |
|merged_pull_requests | Number of merged pull requests |
|open_issues | Number of open issues |
|closed_issues | Number of closed issues |
|forks | Number of forks |
|stargazers | Number of users who "starred" the repo |
|projects | Number of projects (integrated project management tool)|
|size | Size of the source code in KB |
|isOwnerHomepage | Is the name of the repo REPO_OWNER.github.io or REPO_OWNER.github.com? |
|hasHomepage | Does the website REPO_OWNER.github.io/REPO_NAME exist? |
|hasLicense | Does the repo have a license file? |
|hasTravisConfig | Does the repo have a Travis configuration file? |
|hasCircleConfig | Does the repo have a CircleCI configuration file? |
|hasCiConfig | hasTravisConfig OR hasCircleConfig |
|commitsCount | Number of commits |
|branchesCount | Number of branches |
|tagsCount | Number of tags |
|releasesCount | Number of releases |
|LANGUAGE_*| How much code was written in the language in percent (e.g. LANGUAGE_Python, ...)?|

Most of the features were available using the GitHub API.
We added the *isOwnerHomepage* and *hasHomepage* features to detect whether a repository serves its source code using GitHub pages. This could allow us to identify WEB repositories easier.
We furthermore hoped that using *hasCiConfig*, so whether a repo contains a configuration file for a Continuous Integration service, would improve the accuracy of detecting DEV repositories.

#### 3.2.1 Data Cleaning and Preprocessing
Using the GitHub REST API and the GitHub GraphQL API, we were able to receive all features without extensive cleaning or preprocessing of the data.

#### 3.2.2 Feature Selection
Feature Selection describes the process of dropping features that yield no or very little additional information in order to decrease overfitting and accelerate model fitting.
Especially the programming language features needed to be reduced using Feature Selection.  
GitHub detects [over 300 used programming languages](https://github.com/github/linguist/blob/master/lib/linguist/languages.yml) in repositories.
The problem is that a lot of them are used only in a few repositories such that there are a lot of features that only hold very little variance and information.
As an example, among the collected 1400 repositories there were 46 programming languages, like Pony or KiCad, that were only used in one repository at all.
To remove those, we dropped features with low standard deviation and a low overall sum.
As an example, we are already able to drop 135 language features if we require that the sum of a language feature over all 1400 training data entries is supposed to be bigger than 5.


#### 3.2.3 Feature Engineering
In a next step, we derived further features from the features we already collected.
We used polynomial feature generation which takes the input variables and builds all possible polynomial combination of this features up to a given degree.
The idea of taking input features and applying a non-linear method on it to map the original values in another space is called "kernel trick" and is used by Support Vector Machines (SVM) to learn non-linear models as well.  
As an example, suppose a dataset is given with the two features *size* and *watchers*:  

|size|watchers|
|:--:|:------:|
|2|5|
|10|8|

The transformed dataset using polynomial features with a degree up to 2 would look like this:  

|size¹|watchers¹|size¹xwatchers¹|size²|watchers²|
|:---:|:-------:|:-------------:|:---:|:-------:|
|2    |        5|             10|    4|       25|
|10   |        8|             80|  100|       64|

As one can see, the number of generated features increases polynomially in the number of input features. That's why the previous Feature Selection step was very important.

As alternative we could have used deep learning techniques but you need many training samples because of their higher learning complexity. Our ~1500 samples aren't enough for this.
Small feed-forward neural networks are applicable to our problem while deep neural networks are not.

#### 3.2.4 Numeric Metadata Prediction Model
We tried the following classifiers:
...

[Better without language features]

[accuracy from single models]
[accuracy from ensembled models]

#### 3.2.5 Validation of Prediction Model
[accuracy+confusion matrix]

### 3.3 Classification Using Text Data (Description and Readme)
Intuitively, one wouldn't use the numeric features like the number of branches etc. to decide what label fits the repository best.
Instead, one would use the description or the content of the readme to determine it.
For this reason we used term frequency–inverse document frequency (tf-idf) matrices to develop natural language processing (NLP) models that predict the label based on them.
Since there's a semantic difference between the description and the readme of a repository, we discarded the idea of concatenating the text features and training one model on it.
Instead, we trained two seperate models on the description respectively readme of the repositories.

#### 3.3.1 Data Cleaning and Preprocessing
To remove words like 'the', 'a', 'and' etc. that occur very often and yield little meaning, we used the Natural Language Toolkit (NLTK) to drop English stopwords.
Since it's also not important whether the singular or the plural of words are used, we also used this toolkit to stem English words.

#### 3.3.2 Feature Generation from Existing Data

We used a count vectorizer which converts a text into a n-dimensional vector representing the vocabulary, where n is the number of unique words. After this text-to-vector conversion we transformed the vector into a tf-idf vector which is a normalized representation of the original vector.

#### 3.3.3 Feature Selection
[removal of rare/frequent words and stopwords]

#### 3.3.4 Prediction Model

Based on our tf-idf vector we can classify the different repositories using the n-dimensional vector as features and normal classification algorithms.

#### 3.3.5 Validation of Prediction Model
Readme Classifier: 51.6% Accuracy on Validation Data  
![](https://cloud.githubusercontent.com/assets/6676439/21713365/f01abb7e-d3f9-11e6-9a85-6322c7634f1e.png)

Description Classifier: 48.4% Accuracy on Validation Data  
![](https://cloud.githubusercontent.com/assets/6676439/21713307/92e32c2a-d3f9-11e6-91a5-e33192dc58e7.png)

### 3.4 Classification Using Source Code

We tested different approaches to use the source code and connected data of a repository to classify it. For this task we used only data included in the git repository, no github specific data like projects, the wiki pages... Data from the repositories are including source code files with comments and git workflow specific data (branches, commits...).
We used in this chapter mainly the source code, file names and commit messages.

#### 3.4.1 Data Cleaning and Preprocessing

We cloned each repository locally to retrieve the data we need. After this step we were able to merge all non-binary source code files, all filenames and all git commit messages into three different files. We didn't filter based on languages and all UTF-8 files are included. This could be an additional preprocessing step to improve and simplify the stemming and classification.

#### 3.4.2 Feature Generation from Existing Data

We were able to use the same feature generation approach based on the count vectorizer and tf-idf vector as used in the text data classification. [Ugurel et al.](https://clgiles.ist.psu.edu/papers/KDD-2002-whatsthecode.pdf) showed a similar approach successfully.

#### 3.4.3 Feature Selection
[removal of rare/frequent words and stopwords]

#### 3.4.4 Prediction Model

[Same as text data]


#### 3.4.6 Validation of Prediction Model
[accuracy+confusion matrix]

### 3.5 Overall Prediction Model

* [Describe our ensembled model]
* [Document three repositories which work well]

## 4. Implemented Application
Our implemented application takes a file containing GitHub repository URLs, classifies them using an ensemble model that's trained on passed training data and saves the URLs and their computed labels on the disk.
If no training data is given, the input data will be classified using our pre-trained model.
It's possible to pass the input data, which is supposed to have the format of the [challenge example](https://github.com/InformatiCup/InformatiCup2017/blob/master/example-input), using the `-i` argument.
Optional training data can be passed using the `-t` argument.  
As an example, to classify the example data given by the challenge using the training data given by the challenge one would run:  

`python app/main.py -i data/example-input.txt -t data/training_data_small.csv`  

The saved output file `predictions.txt` will have the format of the [challenge example](https://github.com/InformatiCup/InformatiCup2017/blob/master/example-output).  
For setup instructions please refer to the README.md file.

## 5. Validation
* [boolean(!) matrix on validation data]
* [compute recall and precision]
* [discuss quality of results and whether higher yield or higher precision is more important]
* [(elaborate on the additional dataset given by another team?)]


## 6. Extensions
To bring our research work to production, we built a [service](https://git-better.herokuapp.com/) that classifies your public GitHub repositories using models that were trained on our training data.
The server uses GitHub OAuth to authenticate GitHub users and uses their OAuth tokens to request their public repositories and their necessary features.
We're planning to improve the design with visualizations of the repository distribution using D3.  
Another extension would be to recommend [trending GitHub projects](https://github.com/trending) based on the public repositories of the user.  
Since there is no official GitHub API for the trending repositories, we would crawl all websites that are available at https://github.com/trending/{language}?since={since} once a day, where *language* is a supported programming language like Python or Ruby, and *since* is one of 'daily', 'weekly' or 'monthly'.
We would then recommend repositories to the user based on their classified labels, on the preferred language of the user, on the text or even code similarity between the trending projects and those of the user.
To implement the latter one, we could use tf-idf matrices like we already used for the text classifiers.

* [maybe add pictures as fall-back]
