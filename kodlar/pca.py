#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri kümesi
veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:,0:13].values 
y = veriler.iloc[:,13].values 

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#pca dönüşümünden sonra gelen LR
from sklearn.linear_model import LogisticRegression
classifier2= LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)


y_pred= classifier.predict(X_test)
y_pred2= classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
#actual / PCA olmadan çikan sonuç
print('Gercek/ pcasiz karşılaştırma')
cm = confusion_matrix(y_test,y_pred)
print(cm)
#actual / PCA sonrasi Çikan sonuç
print('Gercek/ pcali karşılaştırma')
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)
#PCA sonrasi / PCA öncesi
print('pcali/ pcasiz karşılaştırma')
cm3 = confusion_matrix(y_pred,y_pred2)
print (cm3)
