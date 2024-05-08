#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

#data frame dilimleme (slice)
x= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

# NumPy array (dizi) dönüşümü 
X = x.values
Y = y.values


#linear modelinde görelim import ettik obje oluşturduk ve x den y yi öğrettik
#doğrusal model
from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(X,Y) # x den y yi öğren

#polinomal regression
#doğrusal olmayan model
#2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)# 2 dereceden bir obje oluşturduk
x_poly= poly_reg.fit_transform(X) #x0 x1 x2
#print(x_poly) # görmek için bastırdık 
lin_reg2 =LinearRegression()
lin_reg2.fit(x_poly,Y)

# 4. dereceden polinom 
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3= poly_reg3.fit_transform(X) 
lin_reg3 =LinearRegression()
lin_reg3.fit(x_poly3,Y)

#görselleştirelim
# linear içindi
plt.scatter(X,Y, color= 'red')
plt.plot(x,lin_reg.predict(x),color = 'green')
plt. show()

# polinomal için degree=2 olanın
plt.scatter(X, Y, color= 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color= 'blue')
plt.show()

# polinomal için degree=4 olanın
plt.scatter(X, Y, color= 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color= 'blue')
plt.show()



#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))