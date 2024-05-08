#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kodlar

#veri yükleme
veriler = pd.read_csv('eksikveriler.csv')
#pd.read_csv("veriler.csv")

print(veriler)
boy=veriler[['boy']]
print(boy)
boykilo=veriler[['boy','kilo']]
print(boykilo)


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
    

