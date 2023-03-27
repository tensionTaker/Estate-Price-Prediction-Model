import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn import metrics
from DataClean import Transfer_Data
from DataClean import  Transfer_Target

X = Transfer_Data()
Y = Transfer_Target()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(X_train,Y_train)

def predict_price(location,sqft,bath,bhk):
    try:
        loc_index = np.where(X.columns==location)[0][0]
        x = np.zeros(len(X.columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

        return lr_clf.predict([x])[0]
    except:
        print("Sorry Can't Predict for the given input")

print("Enter the following Details")
print("Location")
location = input()
print("sqft")
sqft = input()
print("bath")
bath = int(input())
print("bhk")
bhk = int(input())

print(predict_price(location,sqft,bath,bhk))





