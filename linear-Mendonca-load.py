from pip._vendor import colorama
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

import numpy as np
import csv

import pickle

x_dates = []
y_prices = []

def load_model(model_name):
    pickle_in = open(model_name,'rb')
    clf = pickle.load(pickle_in )
    return clf


def load_file_graph(file_name):
    with open(file_name, 'r') as csv_file:
        csv_file_reader = csv.reader(csv_file)
        next(csv_file_reader)
        for row in csv_file_reader:
            x_dates.append(int(row[0].split('-')[0]))
            y_prices.append(float(row[1]))


def estimate_coefficients(x,y):

    x_size = np.size(x)
    mean_x, mean_y = np.mean(x), np.mean(y)

    SS_xy = np.sum((y*x) - (x_size*mean_y*mean_x))
    SS_xx = np.sum((x*x) - (x_size*mean_x*mean_x))

    b_1 = SS_xy / SS_xx
    b_0 = mean_y - (b_1*mean_x)
    return (b_0,b_1)


def plot_regression_line(x,y,coefficient_b):

    #plot the pois ate the graph
    plt.scatter(x,y,marker="o",s=30,color="m")

    #predict de responce vector
    y_pred = coefficient_b[0]+ (coefficient_b[1]*x)

    #plotting the regretion line
    plt.plot(x,y_pred,color="g")

    #labels
    plt.xlabel("year")
    plt.ylabel("price")

    plt.show()


load_file_graph('gold_price.csv')


clf = load_model('linearRegre.pickle')

prediction = clf.predict([[2021],[2022],[2023],[2044]] )
print("predict values_________________________")
print(prediction)

x_dates = np.array(x_dates)
y_prices = np.array(y_prices)



b =  estimate_coefficients(x_dates,y_prices)
plot_regression_line(x_dates,y_prices,b)




