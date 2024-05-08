

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#VERİ YÜKLEME
veriler = pd.read_csv('maaslar.csv')
print(veriler)

x= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]
X = x.values
Y = y.values

#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg.predict(X),color="darkblue")

#POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)