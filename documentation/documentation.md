# documentation InformatiCup2017
headlines inspired by KDD process

## data exploration
_In einem ersten Schritt analysieren und dokumentieren Sie die für die Klassifizierung eines Repositories relevanten Mermale (sog.  Features)._
_Erläutern Sie in Ihrer Dokumentation, warum Sie welche Merkmale für die Klassifizierung verwenden und wie Sie Ihr Vorhersagemodell entwickelt haben._

### data retrieval
We used the data provided in the competition desciption and collected more data with different methods:
* Github API for actual data
* github.io pages for WEB

### data analysis
[t-SNE visualisation of data] 
class "DOCS" is clustered, other classes are mixed
validation data does not form a separate cluster to training data, but hard to judge; ~30 vs ~1200 samples
compare with data from other team (4000 samples)
--> all samples (val, our train, other team train) are not bias free [some more thoughts], we should sample randomly from all github repos

--> conclude about further approach

### data selection
_Creating a target data set: selecting a data set, or focusing on a subset of variables, or data samples, on which discovery is to be performed._

### creation of training and test data set
* Split data set into a training set, to train our different classifier, 
a test set, to test the accuracy of our simple classifier, 
a development set, to tune our hyperparameter without overfitting on this level, 
and a evaluation set, to calculate our final accuracy. [Is this conceptual right?]
* Why no k-fold cross validation?
* [we used simple python spliting methods (train_test_split)] 

### classification using numeric metadata of repositories
We used this features:
...

#### data cleaning and preprocessing
...

#### feature generation from existing data
In our previous projects we invested much effort in the manual feature generation with SQL queries etc. or used deep learning techniques to enhance the given data.

This time we do not have the resources/man power to build the features with SQL. So we tried an approach which includes more computing but less human effort.
We used polynomial feature generation which takes the input variables and builds all possible polynomial combination of this features up to a given degree.
[Maybe a simple example]
 
To use deep learning techniques you need many training samples because of their higher learning complexity. Our ~4000 samples aren't enough for this.
Small feed-forward neural networks are applicable to our problem, deep neural networks are not.

### prediction model and automatic classification
We tried the following classifiers:
...

[accuracy from single models]
[accuracy from ensembled models]

#### validation of prediction model
[accuracy+confusion matrix]

### classification using text data (description and readme)

#### data cleaning and preprocessing

#### feature generation from existing data

### prediction model and automatic classification

#### validation of prediction model
[accuracy+confusion matrix]

### classification using source code

#### data cleaning and preprocessing
...

#### feature generation from existing data


### prediction model and automatic classification

#### validation of prediction model
[accuracy+confusion matrix]

## visualization

## general validation of approach
_Berechnen Sie die erzielte  Ausbeute [... und] Präzision_
_Erstellen Sie eine Wahrheitsmatrix in der Sie für diese Repositories sowohl Ihre intuitive Klassifikation, um was für eine Kategorie es sich handelt, als auch die Ergebnisse Ihres automatischen Klassifikators eintragen._