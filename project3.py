import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Calculates the hypothesis for the given order of polynomial
def hypothesis(X, theta, n):
    # initialize hVals as array of zeros, reshape theta to be one bigger than the polynomial order
    hVals = np.zeros((100, 1))
    theta = theta.reshape(1, n+1)

    # range is the number of data points
    for i in range(100):
        xVals = np.zeros(n+1)  # initialize as array of zeros
        for j in range(n+1):
            xVals[j] = pow(X[i], j)  # takes the x value to the power of j

        # reshape the xVals array to correct size, perform matrix multiplication to get the hypothesis
        xVals = xVals.reshape(n+1, 1)
        hVals[i] = np.matmul(theta, xVals)

    hVals = hVals.reshape(100)  # reshape hVals so it's easier to work with
    return hVals


# performs Batch Gradient Descent to optimize the weights
def gradientDescent(X, y, alpha, theta, n, h, iterations):
    # performs loop according to number of selected iterations
    for i in range(iterations):
        # first theta calculation which doesn't contain an x val, 100 is for number of overall data points
        theta[0] = theta[0] - (alpha / 100) * sum(h - y)
        for j in range(1, n+1):  # calculates the rest of the weights for given order
            theta[j] = theta[j] - (alpha / 100) * sum((h - y) * pow(X, j))

        h = hypothesis(X, theta, n)  # calculate new hypothesis

    theta = theta.reshape(1, n+1)  # reshape theta so it's easier to work with
    return theta


# simple mean squared error function
def MSE(y, predictions):
    return (1/100) * sum(np.square(predictions - y))


data = pd.read_csv("synthetic-1.csv")  # change this to use other datasets
Xcsv = data.loc[:, '1']
Ycsv = data.loc[:, '2']
X = np.array(Xcsv)  # put into numpy arrays to make it easy to work with
y = np.array(Ycsv)
alpha = 0.0001
n = 7  # edit the polynomial order here
iterations = 10000  # edit the number of iterations here
theta = np.zeros(n+1)  # initialize theta as an array of zeros

h = hypothesis(X, theta, n)  # calculate initial hypothesis


theta = gradientDescent(X, y, alpha, theta, n, h, iterations)  # perform gradient descent
print(theta)  # outputs updated weights

predictions = hypothesis(X, theta, n)  # gets predictions based on updated weights
print(MSE(y, predictions))  # outputs the mean squared error based on the predictions

# makes the plot
plt.scatter(X, y, label="training data")
plt.plot(X, predictions, label="polynomial regression")
plt.legend()
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Synthetic-1 7th Order')
plt.show()
