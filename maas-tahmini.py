# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 20:37:33 2025

@author: dell
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('maaslary.csv')

X = veriler.iloc[:, [2, 3, 4]].values
Y = veriler.iloc[:, 5].values


#Linear regression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
print('Linear Regression Tahmini:',lin_reg.predict([[10, 100, 1]]))

print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
print(X_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)


print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


sc1 = StandardScaler()
X_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
Y_olcekli = sc2.fit_transform(Y.reshape(-1, 1)).ravel()  


svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_olcekli,Y_olcekli)

tahmin = svr_reg.predict(sc1.transform([[10, 100, 1]]))
tahmin_gercek = sc2.inverse_transform(tahmin.reshape(-1, 1))
print('SVR Tahmini:', tahmin_gercek)


print('SVR R2 degeri')
print(r2_score(Y_olcekli, svr_reg.predict(X_olcekli)))

#Decision Tree Regresyonu
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

print('Decision Tree Tahmini:',r_dt.predict([[10, 100, 1]]))
print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

#Random Forest Regresyonu (Daha İyi Sonuçlar İçin)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(X, Y.ravel()) 

print('Random Forest Tahmini:', rf_reg.predict([[10, 100, 1]]))
print('Random Forest R2 değeri:', r2_score(Y, rf_reg.predict(X)))

#yöntemlerin karşılaştırması


        
