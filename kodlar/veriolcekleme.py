#1.kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.veri önişleme

#2.1veri yükleme
veriler = pd.read_csv('eksikveriler.csv')
#pd.read_csv("veriler.csv")

#test
print(veriler)
boy=veriler[['boy']]
print(boy)
boykilo=veriler[['boy','kilo']]
print(boykilo)

#class mantığını anlamak için önemli değil ve bu kodla alakalı değil
class insan:
    boy =180
    def kosmak(self,b):
        return b+10


ali= insan()
print(ali.boy)
print(ali.kosmak(90))

#eksikveriler
#yasın ortalamayı aldık ve nan yazan kısımlara yazdık 

from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan,strategy='mean')
Yas= veriler.iloc[:,1:4].values
print(Yas)
 

imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)


#encoder Kategorik >> numerik    
ulke=veriler.iloc[:,0:1].values
print(ulke)


from sklearn import preprocessing

le= preprocessing.LabelEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)


ohe = preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray() 

print(ulke)

#numpy dizileri dataframe dönüşümü

sonuc =pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2= pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values

sonuc3= pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiye'])
print(sonuc3)

#dataframe birleştirme

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)


#veri kümesi egitim ve test olarak bölmek

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(s,sonuc3,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)
