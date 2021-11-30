


# Introduction
The problem statement of the project is to develop a **FAKE NEWS DETECTION SYSTEM** using natural language processing and 
Passive Aggressive Classifier, and its accuracy is tested using different machine learning algorithms (_94.16%_) . Detection of false news articles will solve obstacles, the chaos created by fake news. The project uses different widely used machine learning methods (_Logistic Regression, Naive Bayes, Decision Tree, Passive Aggressive Classifier_) and give a response telling the user about the credibility of that news.

# Dataset
 Dataset was taken from kaggle competition - [Fake News](https://www.kaggle.com/c/fake-news/data)
This dataset has a shape of 20800×5. 
train.csv: A full training dataset with the following   attributes:
* id: unique id for a news article
* title: the title of a news article
* author: author of the news article
* text: the text of the article; could be incomplete
* label: a label that marks the article as potentially unreliable:
    1. 1: unreliable
    2. 0: reliable


# Content

```
static
 |-- css
	|-- style.css -> Style Sheet for the HTML page
templates
 |-- main.html -> front end HTML template page 

Fake-News.ipynb  -> Jupyter notebook containg best evaluated Machine Learning model
Fake-news 3 model.ipynb -> Contain 4 ml models - Logistic Regression , Naive Bayes, Decision Tree, Passive Aggressive Classifier
app1.py -> Flask API for receiving the news url, extraction of the news article and returning the prediction 
fake1.py -> Machine Learning Pipeline for classification 
model1.pickle -> Serialized version of Passive Aggressive Classifier model in pickle format
requirements.txt -> Required Libraries 
``` 
# Getting started

#### For installation of the required libraries 
```
python3 -m pip install -r requirements.txt
```
#### Create the machine learning model
```
python3 fake1.py
```
#### Start the FLask API
```
python3 app1.py
```
#### Navigate to the URL 
By default, flask will run on port 5000.
```
http://127.0.0.1:5000
```
<img src="image/fake.gif">
<img src="image/real.gif">

[Thank You!](https://medium.com/analytics-vidhya/building-a-fake-news-classifier-deploying-it-using-flask-6aac31dfe31d)

