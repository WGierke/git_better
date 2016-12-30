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
To get a better idea of how the relationship between the data entries looks like in higher dimensional space, we used PCA and t-SNE to reduce the  complexity of the data to 2D while retaining the principal components respectively the distances between the data points.
The following figure visualizes the distribution of the labeled data entries using t-SNE
![](https://cloud.githubusercontent.com/assets/6676439/21290072/ad44ed02-c4ad-11e6-8314-a078c3b1c853.png)
You can find the complete code to generate the figure in the [t-SNE Visualization Notebook](https://github.com/WGierke/git_better/blob/master/t-SNE%20Visualization.ipynb). [maybe add footnote]
One can notice that the "DOCS" repositories build a cluster while it seems to be more complicated to separate the other classes.  

validation data does not form a separate cluster to training data, but hard to judge; ~30 vs ~1200 samples
compare with data from other team (4000 samples)
--> all samples (val, our train, other team train) are not bias free [some more thoughts], we should sample randomly from all github repos

--> conclude about further approach

## Prediction Model
* document how to avoid overfitting
* explain why we've decided to use the features
* explain how we've developed the prediction model (splitting methods, ensembling etc.)
* reference GitHub REST/GraphQL API, GitHub Search, Google Search, ...

### Data Selection
_Creating a target data set: selecting a data set, or focusing on a subset of variables, or data samples, on which discovery is to be performed._

### Creation of Training and Test Data Set
* Split data set into a training set, to train our different classifier,
a test set, to test the accuracy of our simple classifier,
a development set, to tune our hyperparameter without overfitting on this level,
and a evaluation set, to calculate our final accuracy. [Is this conceptual right?]
* Why no k-fold cross validation?
* [we used simple python spliting methods (train_test_split)]

### Classification Using Numeric Metadata of Repositories
We used this features:
...

#### Data Cleaning and Preprocessing
...

#### Feature Generation from Existing Data
In our previous projects we invested much effort in the manual feature generation with SQL queries etc. or used deep learning techniques to enhance the given data.

This time we do not have the resources/man power to build the features with SQL. So we tried an approach which includes more computing but less human effort.
We used polynomial feature generation which takes the input variables and builds all possible polynomial combination of this features up to a given degree.
[Maybe a simple example]

To use deep learning techniques you need many training samples because of their higher learning complexity. Our ~4000 samples aren't enough for this.
Small feed-forward neural networks are applicable to our problem, deep neural networks are not.

#### Numeric Metadata Prediction Model
We tried the following classifiers:
...

[accuracy from single models]
[accuracy from ensembled models]

##### Validation of Prediction Model
[accuracy+confusion matrix]

### Classification Using Text Data (Description and Readme)

#### Data Cleaning and Preprocessing

#### Feature Generation from Existing Data

#### Prediction Model

##### Validation of Prediction Model
[accuracy+confusion matrix]

### Classification Using Source Code

We test different approaches to use the source code and connected data of a repository to classify it. For this task we use only data included in the git repository, no github specific data like projects, the wiki pages... Data from the repositories are including source code files with comments and git workflow specific data (branches, commits...).
We are using in this chapter mainly the source code, file names and commit messages.

#### Data Cleaning and Preprocessing

We clone each repository locally to retrieve the data we need. After this step we can merge all non-binary source code files, all filenames and all git commit messages into three different files. We don't filter based on languages, all UTF-8 files are included. This could be a additional preprocessing step to improve and simplify the stemming and classification.

#### Feature Generation from Existing Data

**HINT: Text could be used in other chapters as well**

We use a count vectorizer which converts a text into a n-dimensional vector representing the vocabulary, where n is the number of unique words. After this text to vector conversion we transform the vector into a term frequency–inverse document frequency (tf-idf) vector which is a normalized representation of the original vector.

#### Prediction Model

Based on our tf-idf vector we can classify the different repositories using the n-dimensional vector as features and normal classification algorithms.

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

Describe our ensembled model

## Automated Classification
* implement the app that takes the input format and creates the output format
* either 1) prompt for the training data to use or 2) directly include the learned model
* explain the arguments of the main application here while referencing to the manual for setup


## Validation
* boolean(!) matrix on validation data
* compute recall and precision
* discuss quality of results and whether higher yield or higher precision is more important
* (elaborate on the additional dataset given by another team?)


## Extensions
* explain Django app
* reference production instance
* maybe add pictures as fall-back
