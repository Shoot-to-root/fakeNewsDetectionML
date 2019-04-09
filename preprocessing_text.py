# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:07:10 2018

@author: sone_e
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# lower casing...
def lowercase(file):
    file['text'] = file['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return file

# removing punctuation
def remove_punc(file):
    file['text'] = file['text'].str.replace('[^\w\s]','')
    return file

# stop words removal
def remove_stopwords(file):
    stop = stopwords.words('english')
    file['text'] = file['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop and not x.isdigit()))
    return file
    
#lemmatization
def lemmatization(file):
    lemmer=WordNetLemmatizer()
    file['text'] = file['text'].apply(lambda x: " ".join([lemmer.lemmatize(word) for word in x.split()]))
    return file
  
# tokenization
def tokenization(text): 
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens

def preprocessing(file):
    lowercase(file)
    remove_punc(file)
    remove_stopwords(file)
    lemmatization(file)
    return file