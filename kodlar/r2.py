
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

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

print("Linear  r2 değeri")
print(r2_score(Y, lin_reg.predict(X)))

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

print("Polinomal  r2 değeri")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))



#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler() 
y_olcekli = sc2.fit_transform(Y)


# destek vektör makineleri
from sklearn.svm import SVR

svr_reg= SVR(kernel='rbf')
#svr_reg= SVR(kernel ='sigmoid')
#svr_reg= SVR(kernel ='poly')
#svr_reg= SVR(kernel ='precomputed')

svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color ='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

print("DVM R2 değeri")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))



#Desicion Tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X -0.4

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")
plt.plot(X,r_dt.predict(K),color="green")
plt.plot(X,r_dt.predict(Z),color="yellow")

plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

print("Decision Tree R2 değeri")
print(r2_score(Y, r_dt.predict(X)))


#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0) #n_estimators kaç decision tree çizilecek

rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")
plt.plot(X,rf_reg.predict(K),color="green")
plt.plot(X,rf_reg.predict(Z),color="yellow")





#r2_score(y_true, y_pred) #y gercek değeri, y tahmin değeri

print("Random Forest R2 değeri")
print(r2_score(Y, rf_reg.predict(X)))

print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))



#Özet r2 değerleri

print("----------------------------------------")

print("Linear  r2 değeri")
print(r2_score(Y, lin_reg.predict(X)))


print("Polinomal  r2 değeri")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


print("DVM R2 değeri")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


print("Decision Tree R2 değeri")
print(r2_score(Y, r_dt.predict(X)))

print("Random Forest R2 değeri")
print(r2_score(Y, rf_reg.predict(X)))
