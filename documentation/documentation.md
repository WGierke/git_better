# InformatiCup2017 Documentation
## Challenge Description
This years InformatiCup challenge was to classify GitHub repositories automatically based on given class descriptions and sample data. In this work we present how we explored the given data, detected relevant features and built an application that predicts repository labels using different machine learning algorithms.


## Data Exploration
* (visualization)
* analyze and document relevant features
* (explain data cleaning and preprocessing)

### Data Retrieval
We used the data provided in the competition desciption and collected more data with different methods:
* Github API for actual data
* github.io pages for WEB

### Data Analysis
[t-SNE visualisation of data] 
class "DOCS" is clustered, other classes are mixed
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

#### Data Cleaning and Preprocessing
...

#### Feature Generation from Existing Data

#### Prediction Model

##### Validation of Prediction Model
[accuracy+confusion matrix]

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
