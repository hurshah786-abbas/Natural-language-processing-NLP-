# -*- coding: utf-8 -*-

#importing the labraries

import numpy as np
import re
import pickle
import nltk

from nltk.corpus import stopwords
from sklearn.datasets import load_files

#importing dataSet
reviews = load_files('txt_sentoken')
X,y = reviews.data,reviews.target

#storing as pickle files 
with open('X,pickle','wb') as f:
    pickle.dump(X,f)
    
with open ('y,pickle','wb') as f:
    pickle.dump(y,f)
    
#unpickling the data set
with open('X,pickle','rb') as f:
    pickle.load(f)
    
with open ('y,pickle','rb') as f:
    pickle.load(f)    
    
#creating corpus
corpus = [] 
for i in range (0,len(X)):
    review = re.sub(r'\W',' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'^[a-z]\s+',' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000,min_df = 3 , max_df =0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000,min_df = 3 , max_df =0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test = train_test_split(X,y,test_size = 0.2 ,random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

sent_pred = classifier.predict(text_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)


#pickling the classifier
with open ('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
#pickling the vectorizier
with open ('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
#unpickling the classifier
with open ('classifier.pickle','rb') as f:
    clf = pickle.load(f)

with open ('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
        
sample = ["""Taqi is a good person he is good in cricket as well as in communication """]
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))     


sample1 = ["""Protect your good image from the eyes of negative viewers, who may look at your good appearance with an ugly fiendish eye, and ruin your positive qualities with their chemical infested tongues."""]
sample1 = tfidf.transform(sample1).toarray()
print(clf.predict(sample1))  


sample2 = ["""Taqi is good person he is moderate in cricket as well as in communication """]
sample2 = tfidf.transform(sample2).toarray()
print(clf.predict(sample2))      