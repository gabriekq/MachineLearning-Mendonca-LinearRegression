
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

import numpy as np
import csv

import pickle

#Read Data
x_dates = []
y_prices = []

def read_data(file_name):
    with open(file_name,'r') as csv_file:
         csv_file_reader = csv.reader(csv_file)
         next(csv_file_reader)
         for row in csv_file_reader:
             x_dates.append(int(row[0].split('-')[0]))
             y_prices.append(float(row[1]))



def save_model(model_name,clf_name):
    with open(model_name,'wb') as f:
        pickle.dump(clf_name,f)





read_data("gold_price.csv")

x_dates = np.array(x_dates)
y_prices = np.array(y_prices)



x_train, x_test, y_train, y_test = train_test_split(x_dates,y_prices,test_size=0.2)

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

y_test = y_test.reshape(-1,1)

y_train = y_train.reshape(-1,1)

ctf = LinearRegression(n_jobs=-1)
ctf.fit(x_train,y_train)


print("datas")
print(x_train) #Datas
print("\n")
print("\n")
print(y_train) #preços

accuracy = ctf.score(x_test,y_test)
print("accuracy",accuracy)


save_model(model_name='linearRegre.pickle',clf_name=ctf)


