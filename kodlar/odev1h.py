

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)


#encoder: Kategorik -> Numeric
from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c=veriler2.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)


havadurumu = pd.DataFrame(data=c, index = range(14), columns = ['o','r','s'])
sonveriler=pd.concat([havadurumu,veriler.iloc[:,1:3]], axis=1)
print(sonveriler)
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler], axis=1)
print(sonveriler)



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression

regessor = LinearRegression()
regessor.fit(x_train, y_train)

y_pred = regessor.predict(x_test)
print(y_pred)


#gerieleme
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)


X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l =np.array(X_l,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

sonveriler =sonveriler.iloc[:,1:]

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)


X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l =np.array(X_l,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())


#p degeri en yüksek olan ı cıkardık ve yeniden tahmin ettik
x_train= x_train.iloc[:,1:]
x_test= x_test.iloc[:,1:]


regessor = LinearRegression()
regessor.fit(x_train, y_train)

y_pred = regessor.predict(x_test)
