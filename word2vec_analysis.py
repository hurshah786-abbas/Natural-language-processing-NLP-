# -*- coding: utf-8 -*-

import nltk
import urllib
import bs4 as bs
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec

source = urllib.request.urlopen('https://www.gisma.com/school/about-us').read()

soup = bs.BeautifulSoup(source,'lxml')

text = "" 
for paragraph in soup.find_all('p'):
    text += paragraph.text

text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub('[@#\$%&\*\(\)\<\>\?\'\":;\]\[-]',' ',text)
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

model = Word2Vec(sentences,min_count=1)

word = model.wv.vocab

vector = model.wv['Global']

similar = model.wv.most_similar('Global')