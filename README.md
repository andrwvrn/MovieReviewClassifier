# MovieReviewClassifier

This project contains a model and a web application for movie reviews classification. The model returns predictions for a rating of a given review and also assigns it a class('positive' or 'negative').

***

# Installation

Clone repo:

```sh
$ git clone https://github.com/andrwvrn/MovieReviewClassifier.git
```

##### Set and activate new environment:

Using [conda](https://docs.conda.io/en/latest/):
```sh
$ conda create -n new_env
$ conda activate new_env
```
Using [venv](https://docs.python.org/3/library/venv.html):
```sh
$ python3 -m venv new_env
$ source new_env/bin/activate
```
##### Install required packages and dependencies:
`cd` to the project folder:
```sh
$ cd MovieReviewClassifier
```
Run installation:
```sh
$ pip install -r requirements.txt
```
If you want to manage web application locally you will need to additionally install [Django](https://docs.djangoproject.com/en/3.0/intro/install/).

# Making classifications
There are two ways to get classification of a review by MovieReviewClassifier. 
1. You can do it locally by using script `classify.py` placed in the root directory of a repo. There is an example of classification with the sample of a review extracted from the test part of our dataset and placed in the `/sample_reviews` folder.
```sh
$ python3 classify.py sample_reviews/sample_pos_review.txt
Rating: 7 Class: Positive
```
2. The second way is to use Django application deployed on [Heroku](https://www.heroku.com/). Here is an example of code from Python interpreter but you can also put it in your `.py` script and run:
```python
>>>import requests
>>>req = requests.get('https://movierev.herokuapp.com/cls', {'text':'This films is good.'})
>>>print(req.content)
b'{"prediction": {"Rating": 7, "Class": "Positive"}}'
```
Server returns `JsonResponse` class and to get values from it you can do next:
```python
>>>import json
>>>req_d = json.loads(req.content)
>>>print(req['prediction'])
{'Rating': 7, 'Class': 'Positive'}
```
# Data
Data was taken from [Large Movie Review Dataset v1.0](https://ai.stanford.edu/~amaas/data/sentiment/). This dataset contains movie reviews along with their associated binary sentiment polarity labels. The core dataset contains 50,000 reviews split evenly into 25k train and 25k test sets. The overall distribution of labels is balanced (25k pos and 25k neg). In the labeled train/test sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included in the train/test sets. According to this, the classifier gives rating predictions for 'negative' reviews from 1 to 4 and for 'positive' reviews from 7 to 10. There are 8 classes overall.

# Data preparation
At first step files from [Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) have been read and stored in two `pandas.DataFrame`s. The first `pandas.DataFrame` contains `train` data and the second contains `test` data. They can be represented as tables:
|      |Review     |Rating                                            |
|------|---------------------------------------------------------|----|
|0     |This is fantastic! Everything from the Score -...        |10  |
|1     |This movie was amazing!!!! From beginning to e...	 |10  |
|2     |The first time I've seen this DVD, I was not o...	 |10  |
|3     |One of the flat-out drollest movies of all-tim...	 |10  |
|4     |When I first got wind of this picture, it was ...	 |9   |

There were duplicated reviews in the `'Review'` column of our tables and after deleting them `24904` and `24801` rows left in `train` and `test` tables correspondingly. For convenience and succeeding manipulations these `pandas.DataFrame`s have been saved as `.csv` files to `/csv` folder of the repo.

# Text processing
Reviews processing was provided in 3 steps:
1. Deriving words with length more than 3 letters using `re` module with regular expression `[a-zA-Z]{3,}`.
2. Words [stemming](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) using `nltk.SnowballStemmer`
3. Deleting *stopwords* using `nltk.stopwords`

After these steps our data looks like:
| |Review	                                           | Rating |review_parse                                |
|-|--------------------------------------------------------|--------|--------------------------------------------|
|0|	This is fantastic! Everything from the Score -...  |10|	fantast everyth score final credit role movi m...|
|1|	This movie was amazing!!!! From beginning to e...  |10|	movi amaz begin end movi pack fun laugh music ...|
|2|	The first time I've seen this DVD, I was not o...  |10|	first time seen dvd onli happi becaus fact fir...|
|3|	One of the flat-out drollest movies of all-tim...  |10|	one flat drollest movi time sim rutherford bes...|
|4|	When I first got wind of this picture, it was ...  |9 |	first got wind pictur call shepherd suppos fil...|

Column `'review_parse'` represents processed text of a review. New `pd.DataFrame`s with cleaned and processed data have been stored to `/data` folder in `.pkl` format.

# Model selection
To make features from the dataset we used `TfidfVectorizer` from `scikit-learn` library. After deriving features *cross-validation* was performed on the train data splitting it in 5 folds with four different models. These models are:
- RandomForestClassifier
- LinearSVC
- MultinomialNB
- LogisticRegression

`OneVsOne` multiclass strategy was used for training. Then the mean value of `accuracy_score` was counted for each model.

![Model selection](https://github.com/andrwvrn/MovieReviewClassifier/raw/master/images/cv.png)

As we can see `LinearSVC` and `Logistic Regression` perform better than the other two classifiers, with `Logistic Regression` having a slight advantage thus it has been chosen for subsequent utilization.

# Hyperparameters tuning
For hyperparameters tuning [W&B](https://www.wandb.com/) platform has been used and a *random search* with *cross-validation* on *3* folds was performed. File `sweep.yaml` with hyperparameters values and a script for `wandb` sweep agent are stored in `hparams` folder.

![hparams search](https://github.com/andrwvrn/MovieReviewClassifier/raw/master/images/hparams.png)

Best parameters gave us an `accuracy_score` of `0.4297` but parameters with a slightly less score of `0.4278` were chosen in order to make final model more lightweight. These parameters are:
```yaml
TfidfVectorizer:
  max_df: 0.95
  ngram_range: (1,1)
LogisticRegression:
  penalty: l2
  solver: liblinear
  C: 2
```

# Model testing
Model performance has been tested on the test set.

![Confusion matrix](https://github.com/andrwvrn/MovieReviewClassifier/raw/master/images/cmatrix.png)

We can see that our model makes good predictions for marginal cases but it is not very effective in segregating classes inside `positive` or `negative` groups.

# Notebooks
To repeat above steps by yourself you can run `.ipynb` notebooks from `/notebooks` folder in order:
- `read_files.ipynb`
- `preprocessing.ipynb`
- `training.ipynb`
