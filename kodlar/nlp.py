# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:23:37 2024

@author: Oguzhan
"""
import numpy as np
import pandas as pd

import re

r = open('Restaurant_Reviews.csv', 'r')
w = open('Restaurant_Reviews_No_Commas.csv', 'w')
w.write(r.readline()) #ilk satırı yaz
skipfirstline = r.readlines()[0:]
for line in skipfirstline:
    line = re.sub(',', '',line)
    line = re.sub('"','',line)
    line = line[:-2]+','+line[-2:]
    #print(line)
    w.write(line)
r.close()
w.close()

import nltk

from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords
#preprocessing (ön işleme)
derlem =[]
for i in range(1000):
    yorumlar = pd.read_csv('Restaurant_Reviews_No_Commas.csv')
    yorum = re.sub('[^a-zA-Z]'," ",yorumlar['Review'][i])
    yorum = yorum.lower() 
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)

# Feature Extraction (öznitelik çıkarımı)
# Bag Of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)

X= cv.fit_transform(derlem).toarray() #bağımsız değişken
y= yorumlar.iloc[:,1].values #bağimli değişken

#Makine öğrenmesi
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train, y_train)
y_pred= gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test,y_pred)
print(cm) #72.5 accuray


