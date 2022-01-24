# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial.polynomial import polyfit
from sklearn.linear_model import LinearRegression


noIterations = 1500
alpha = 0.01
url = "data.csv"
dataset = pd.read_csv(url)
samples= len(dataset)
population=dataset['population']
profit=dataset['profit']
listIteration=[]
listError=[]

def predict(X,theta0,theta1):
    yPredict = (theta1*X) + theta0
    return yPredict

def gradientDescent():
    theta0 = 0
    theta1 = 0
    for x in range(noIterations):
       yPredicted = predict(population, theta0, theta1)
       totalError= (1/2*samples)*sum(pow((yPredicted-profit),2))
       theta0=theta0-((alpha/samples)*(sum((yPredicted-profit)*1)))
       theta1=theta1-((alpha/samples)*(sum((yPredicted-profit)*population)))
       listIteration.append(x)
       listError.append(totalError)
       print("ERROR in iteration " , x, " ",totalError)
    print("Equation: Y = " , theta0 ,"+", theta1,"x" )
    plot1(theta0,theta1)
    plot2(listError,listIteration)

def plot1(theta0,theta1):
    x = np.array(population)
    y = np.array(profit)
    plt.plot(x, y, 'o')
    plt.plot(x, theta1 * x + theta0 ,color='red')
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.show()


def plot2(listError, listIteration):
    plt.xlabel('No of Iterations')
    plt.ylabel('Error')
    x=np.array(listIteration)
    y = np.array(listError)
    plt.plot(x,y,color='red')
    plt.show()

gradientDescent()

