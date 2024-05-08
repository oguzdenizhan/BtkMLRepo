

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')

#print(veriler)


#encoder: Kategorik -> Numeric
hava = veriler.iloc[:,0:1].values
#print(hava)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

hava[:,0] = le.fit_transform(veriler.iloc[:,0])

#print(hava)


ohe = preprocessing.OneHotEncoder()
hava = ohe.fit_transform(hava).toarray()
#print(hava)

#encoder: Kategorik -> Numeric
windy = veriler.iloc[:,3:4].values
#print(windy)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,-1] = le.fit_transform(veriler.iloc[:,3:4])

#print(windy)



ohe = preprocessing.OneHotEncoder()
windy = ohe.fit_transform(windy).toarray()
#print(windy)

windy= windy[:,0]

#print(windy)

#encoder: Kategorik -> Numeric
ply = veriler.iloc[:,4:5].values
print(ply)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ply[:,-1] = le.fit_transform(veriler.iloc[:,4:5])

print(ply)

ohe = preprocessing.OneHotEncoder()
ply = ohe.fit_transform(ply).toarray()
print(ply)

#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=hava, index = range(14), columns = ['overcast','rainy','sunny'])
print(sonuc)

th= veriler.iloc[:,1:3].values

sonuc2 = pd.DataFrame(data=th, index = range(14), columns = ['temperature','humidity'])
print(sonuc2)

sonuc3 = pd.DataFrame(data=windy, index = range(14), columns = ['windy'])
print(sonuc3)



sonuc4 = pd.DataFrame(data = ply[:,:1], index = range(14), columns = ['play'])
print(sonuc4)



#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

s3=pd.concat([s2,sonuc4], axis=1)
print(s3)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s2,sonuc4,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression

regessor = LinearRegression()
regessor.fit(x_train, y_train)

y_pred = regessor.predict(x_test)





#gerieleme
import statsmodels.api as sm

oyna= s3.iloc[:,5:6].values
print(oyna)

X = np.append(arr = np.ones((14,1)).astype(int),values=s2,axis=1)


X_l = s2.iloc[:,[0,1,2,3,4,5]].values
X_l =np.array(X_l,dtype=float)
model=sm.OLS(oyna,X_l).fit()
print(model.summary())

X_l = s2.iloc[:,[0,1,3,4,5]].values
X_l =np.array(X_l,dtype=float)
model=sm.OLS(oyna,X_l).fit()
print(model.summary())

X_l = s2.iloc[:,[0,1,3,5]].values
X_l =np.array(X_l,dtype=float)
model=sm.OLS(oyna,X_l).fit()
print(model.summary())

X_l = s2.iloc[:,[0,3,5]].values
X_l =np.array(X_l,dtype=float)
model=sm.OLS(oyna,X_l).fit()
print(model.summary())

X_l = s2.iloc[:,[3,5]].values
X_l =np.array(X_l,dtype=float)
model=sm.OLS(oyna,X_l).fit()
print(model.summary())
