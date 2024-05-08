

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('maaslar.csv')

x= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

X = x.values
Y = y.values
#linear modelinde görelim import ettik obje oluşturduk ve x den y yi öğrettik
from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(X,Y) # x den y yi öğren

#görselleştirelim
plt.scatter(X,Y, color= 'red')
plt.plot(x,lin_reg.predict(x),color = 'green')
plt. show()


#polinomal regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)# 2 dereceden bir obje oluşturduk
x_poly= poly_reg.fit_transform(X) #x0 x1 x2
print(x_poly)

lin_reg2 =LinearRegression()
lin_reg2.fit(x_poly,Y)

plt.scatter(X, Y, color= 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color= 'blue')
plt.show()



poly_reg = PolynomialFeatures(degree=4)
x_poly= poly_reg.fit_transform(X) 
print(x_poly)

lin_reg2 =LinearRegression()
lin_reg2.fit(x_poly,Y)

plt.scatter(X, Y, color= 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color= 'blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))