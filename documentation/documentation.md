# InformatiCup2017 Documentation
## Challenge Description
This years InformatiCup challenge was to classify GitHub repositories automatically based on given class descriptions and sample data.
In this work we present how we explored the given data, detected relevant features and built an application that predicts repository labels using different machine learning algorithms.


## Data Exploration
This section explains how we extended the training data set and how we explored it using different dimension reduction algorithms and visualization tools.

### Data Retrieval
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

### Data Analysis
To get a better idea of how the relationship between the data entries looks like in a higher dimensional space, we used PCA and t-SNE to reduce the  complexity of the data to 2D while retaining the principal components respectively the distances between the data points.
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

## Prediction Model
* [document how to avoid overfitting]
* [explain why we've decided to use the features]
* [explain how we've developed the prediction model (splitting methods, ensembling etc.)]
* [reference GitHub REST/GraphQL API, GitHub Search, Google Search, ...]

### Data Selection
[_Creating a target data set: selecting a data set, or focusing on a subset of variables, or data samples, on which discovery is to be performed._]

### Creation of Training and Test Data Set
* [Split data set into a training set, to train our different classifier,
a test set, to test the accuracy of our simple classifier,
a development set, to tune our hyperparameter without overfitting on this level,
and a evaluation set, to calculate our final accuracy. [Is this conceptually right?]]
* [Why no k-fold cross validation?]
* [we used simple python splitting methods (train_test_split)]

### Classification Using Numeric Metadata of Repositories
To develop classifiers based on numeric metadata of repositories, we used the following features:

|Feature Name|Description|
|------------|-----------|
|watchers | Number of users who watch the repo |
|mentionableUsers | Number of users who contributed to the repo |
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

#### Data Cleaning and Preprocessing
Using the GitHub REST API and the GitHub GraphQL API, we were able to receive all features without extensive cleaning or preprocessing of the data.

#### Feature Selection
Feature Selection describes the process of dropping features that yield no or very little additional information in order to decrease overfitting and accelerate model fitting.
Especially the programming language features needed to be reduced using Feature Selection.  
GitHub detects [over 300 used programming languages](https://github.com/github/linguist/blob/master/lib/linguist/languages.yml) in repositories.
The problem is that a lot of them are used only in a few repositories such that there are a lot of features that only hold very little variance and information.
As an example, among the collected 1400 repositories there were 46 programming languages, like Pony or KiCad, that were only used in one repository at all.
To remove those, we dropped features with low standard deviation and a low overall sum.
As an example, we are already able to drop 135 language features if we require that the sum of a language feature over all 1400 training data entries is supposed to be bigger than 5.


#### Feature Engineering
In a next step, we derived further features from the features we already collected.
We used polynomial feature generation which takes the input variables and builds all possible polynomial combination of this features up to a given degree.
The idea of taking input features and applying a non-linear method on it to map the original values in another space is called "kernel trick" and is used by Support Vector Machines to learn non-linear models as well.  
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

To use deep learning techniques you need many training samples because of their higher learning complexity. Our ~1500 samples aren't enough for this.
Small feed-forward neural networks are applicable to our problem while deep neural networks are not.

#### Numeric Metadata Prediction Model
We tried the following classifiers:
...

[Better without language features]

[accuracy from single models]
[accuracy from ensembled models]

##### Validation of Prediction Model
[accuracy+confusion matrix]

### Classification Using Text Data (Description and Readme)

#### Data Cleaning and Preprocessing

#### Feature Generation from Existing Data

We used a count vectorizer which converts a text into a n-dimensional vector representing the vocabulary, where n is the number of unique words. After this text-to-vector conversion we transformed the vector into a term frequency–inverse document frequency (tf-idf) vector which is a normalized representation of the original vector.

#### Prediction Model

Based on our tf-idf vector we can classify the different repositories using the n-dimensional vector as features and normal classification algorithms.

##### Validation of Prediction Model
[accuracy+confusion matrix]

### Classification Using Source Code

We tested different approaches to use the source code and connected data of a repository to classify it. For this task we used only data included in the git repository, no github specific data like projects, the wiki pages... Data from the repositories are including source code files with comments and git workflow specific data (branches, commits...).
We used in this chapter mainly the source code, file names and commit messages.

#### Data Cleaning and Preprocessing

We cloned each repository locally to retrieve the data we need. After this step we were able to merge all non-binary source code files, all filenames and all git commit messages into three different files. We didn't filter based on languages, all UTF-8 files are included. This could be a additional preprocessing step to improve and simplify the stemming and classification.

#### Feature Generation from Existing Data

We were able to use the same feature generation approach based on the count vectorizer and tf-idf vector used in the text data classification. [Ugurel et al.](https://clgiles.ist.psu.edu/papers/KDD-2002-whatsthecode.pdf) showed a similar approach succesfully.

#### Prediction Model

[Same as text data]


##### Validation of Prediction Model
[accuracy+confusion matrix]

### How to Overcome Overfitting

When we trained the classifiers on the collected training data and validated them on the given validation data, we noticed very early that all classifiers overfit heavily.
On average, they yielded an accuracy of 80% on the training data but only 20% on the validation data.
To overcome this big overfitting problem, we used several approaches.
- Firstly, we tried to automatically collect as much labeled data as possible.
- Secondly, we use model ensembling so multiple models can learn distinct from each other how to separate the classes.
In the end, the models weighted predictions are aggregated to one prediction.
- As a last approach, we're using hyperparameter tuning using Grid search to find the best parameter set for each classifier such that each one generalizes as much as possible.

### Overall Prediction Model

[Describe our ensembled model]

## Automated Classification
* [implement the app that takes the input format and creates the output format]
* [either 1) prompt for the training data to use or 2) directly include the learned model]
* [explain the arguments of the main application here while referencing to the manual for setup]


## Validation
* [boolean(!) matrix on validation data]
* [compute recall and precision]
* [discuss quality of results and whether higher yield or higher precision is more important]
* [(elaborate on the additional dataset given by another team?)]


## Extensions
To bring our research work to production, we built a [service](https://git-better.herokuapp.com/) that classifies your public GitHub repositories using models that were trained on our training data.
The server uses GitHub OAuth to authenticate GitHub users and uses their OAuth tokens to request their public repositories and their necessary features.
We're planning to improve the design with visualizations of the repository distribution using D3.  
Another extension would be to recommend [trending GitHub projects](https://github.com/trending) based on the public repositories of the user.
Since there is no official GitHub API for the trending repositories, we would crawl all websites that are available at https://github.com/trending/{language}?since={since} once a day, where *language* is a supported programming language like Python or Ruby, and *since* is one of 'daily', 'weekly' or 'monthly'.
We would then recommend repositories to the user based on their classified labels, on the preferred language of the user, on the text or even code similarity between the trending projects and those of the user.
To implement the latter one, we could use tf-idf matrices like we already used for the text classifiers.

* [maybe add pictures as fall-back]
