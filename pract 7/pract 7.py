# -*- coding: utf-8 -*-
"""
Created on Sun May 22 01:15:12 2022

@author: technOrbit
"""

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')

#Sentence Tokenization: 
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome. The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)

#Word Tokenization:
tokenized_word=word_tokenize(text)
print(tokenized_word)

#.Frequency Distribution:
fdist = FreqDist(tokenized_word)
print(fdist)

#Frequency Distribution Plot
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()


#POS Tagging:
sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nltk.word_tokenize(sent)
print(tokens)

nltk.download('averaged_perceptron_tagger') 
nltk.pos_tag(tokens)

#Stopwords: 
from nltk.corpus import stopwords 
stop_words=set(stopwords.words("english")) 
print(stop_words)

#.Removing Stop words: 
tokenized_sent=nltk.word_tokenize(sent)
filtered_sent=[]
for w in tokenized_sent:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_sent)
print("Filterd Sentence:",filtered_sent)


# Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
 stemmed_words.append(ps.stem(w))
print("Filtered Sentence:",filtered_sent) 
print("Stemmed Sentence:",stemmed_words)

#Lexicon Normalization
#performing stemming and Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()
word = "flying" 
print("Lemmatized Word:",lem.lemmatize(word,"v")) 
print("Stemmed Word:",stem.stem(word))


import pandas as pd
import sklearn as sk
import math

first_sentence = "Data Science is the sexiest job of the 21st century"
second_sentence = "machine learning is the key for data science"

#split so each word have their own string
first_sentence = first_sentence.split(" ")
second_sentence = second_sentence.split(" ")

#join them to remove common duplicate words
total=set(first_sentence).union(set(second_sentence))
print(total)

wordDictA = dict.fromkeys(total, 0)
wordDictB = dict.fromkeys(total, 0)
for word in first_sentence:
 wordDictA[word]+=1
 
for word in second_sentence:
 wordDictB[word]+=1

print(pd.DataFrame([wordDictA, wordDictB]))

def computeTF(wordDict, doc):
 tfDict = {}
 corpusCount = len(doc)
 for word, count in wordDict.items():
     tfDict[word] = count/float(corpusCount)
 return(tfDict)

#running our sentences through the tf function:
tfFirst = computeTF(wordDictA, first_sentence)
tfSecond = computeTF(wordDictB, second_sentence)

#Converting to dataframe for visualization
tf = pd.DataFrame([tfFirst, tfSecond])

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in wordDictA if not w in stop_words]
print(filtered_sentence)

def computeIDF(docList):
 idfDict = {}
 N = len(docList)
 
 idfDict = dict.fromkeys(docList[0].keys(), 0)
 for word, val in idfDict.items():
     idfDict[word] = math.log10(N / (float(val) + 1))
 
 return(idfDict)#inputing our sentences in the log file
idfs = computeIDF([wordDictA, wordDictB])

def computeTFIDF(tfBow, idfs):
 tfidf = {}
 for word, val in tfBow.items():
     tfidf[word] = val*idfs[word]
 return(tfidf)
#running our two sentences through the IDF:
idfFirst = computeTFIDF(tfFirst, idfs)
idfSecond = computeTFIDF(tfSecond, idfs)
#putting it in a dataframe
idf= pd.DataFrame([idfFirst, idfSecond])
print(idf)

#first step is to import the library
from sklearn.feature_extraction.text import TfidfVectorizer 
#for the sentence, make sure all words are lowercase or you will run 
#into error. for simplicity, I just made the same sentence all 
#lowercase
firstV= "Data Science is the sexiest job of the 21st century" 
secondV= "machine learning is the key for data science"
#calling the TfidfVectorizer 
vectorize= TfidfVectorizer() 
#fitting the model and passing our sentences right away:
response= vectorize.fit_transform([firstV, secondV])
print(response)
