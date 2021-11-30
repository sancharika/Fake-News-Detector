#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv('train.csv')
df.head()
df = df.fillna(' ')
df.label = df.label.astype(str)
df.label = df.label.str.strip()

X = df['text']
y = df['label']

# count_vectorizer = CountVectorizer()
# count_vectorizer.fit_transform(X)

# freq_term_matrix = count_vectorizer.transform(X)
# tfidf = TfidfTransformer(norm = "l2")
# tfidf.fit(freq_term_matrix)
# tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)


#Splitting the data into train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Creating a pipeline that first creates bag of words(after applying stopwords) & then applies Multinomial Naive Bayes model
# pipeline = Pipeline([('tfidf', TfidfTransformer(stop_words='english')),
#                     ('classifier', PassiveAggressiveClassifier(max_iter=50))])
pipeline=Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('classifier', PassiveAggressiveClassifier(max_iter=50)),
                    ])

# #Training our data
# pipeline.fit(X_train, y_train)

# #Predicting the label for the test data
# pred = pipeline.predict(X_test)

# pac=PassiveAggressiveClassifier(max_iter=50)

pipeline.fit(X_train,y_train)

 #Predict on the test set and calculate accuracy

pred=pipeline.predict(X_test)


#Checking the performance of our model
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

#Serialising the file
with open('model1.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)

