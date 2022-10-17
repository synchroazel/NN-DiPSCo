# -*- coding: utf-8 -*-

""" Linear Regression as Neural Network """

import numpy as np
from matplotlib import pyplot as plt


class NN_linreg:
    def __init__(self):
        self.loss = []

    def linear(self, x):
        return x

    def linear_prime(self, x):
        return 1

    def Cost(self, yhat, y):  # mean square error
        return np.mean((yhat - y) ** 2)

    def Cost_prime(self, yhat, y):
        return np.atleast_2d(2 * (yhat - y))

    def train(self, x, y, lr=.1, epochs=100):
        # Determine the number of features
        nfeat = x.shape[1] if x.ndim > 1 else 1

        # Initialize the slope (weight) and the intercept (bias)
        w1 = np.random.rand(nfeat, h_neurons)
        b = np.random.rand(h_neurons)

        w2 = np.random.rand(h_neurons, o_neurons)
        b = np.random.rand(o_neurons)

        # Start training
        for e in range(epochs):
            # =======
            # FORWARD
            # =======

            # Compute z
            z = np.matmul(x, w) + b

            # Apply activation function
            a = self.linear(z)

            # Compute Cost 
            self.loss.append(self.Cost(a, y))

            # ========
            # BACKWARD
            # ========

            # Compute gradients of each step
            δC_δa = self.Cost_prime(a, y)
            δa_δz = self.linear_prime(z)
            δz_δw = x
            δz_δb = 1

            # Apply chain rule
            δC_δw = np.dot(δz_δw.T, δC_δa * δa_δz)
            δC_δb = np.dot(δz_δb, δC_δa * δa_δz).sum()

            # δC_δw = np.sum(x*(2*(a - y)),axis=0)
            # δC_δb = np.mean((2*(a - y)),axis=0)

            # Update parameters
            w -= (lr * δC_δw) / len(x)
            b -= (lr * δC_δb) / len(x)

            print('Epoch: %d  |  Loss: %1.8f' % (e + 1, self.loss[-1]))

        # Store parameters as attributes    
        self.w = w
        self.b = b

    def predict(self, x):
        return np.dot(x, self.w) + self.b

    def plot_loss(self):
        plt.plot(self.loss, c="r")
        plt.title('Loss across the epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Import regression dataset
data = datasets.load_boston()
x = np.atleast_2d(data.data[:, :])
y = np.atleast_2d(data.target).T

# Normalize data
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# Fit Neural Network
nn = NN_linreg()
nn.train(x, y, lr=.1, epochs=100)
nn.plot_loss()

# Fit Linear model
linreg = LinearRegression()
linreg.fit(x, y)

# Compare parameters
print('''
Neural Network 
weights:    %s
bias:       [%f]

Linear Model
slopes:     %s
intercept:  [%f]
''' % (' '.join(map(str, nn.w.T)), nn.b, ' '.join(map(str, linreg.coef_)), linreg.intercept_[0]))

# Compare mse
nn_mse = np.mean((nn.predict(x) - y) ** 2)
linreg_mse = np.mean((linreg.predict(x) - y) ** 2)
print('''
Mean Square Error
Neural Network        %f
Linear Regression     %f
''' % (nn_mse, linreg_mse))

# Compare predictions with a scatter plot
plt.title('Predictions')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x, y)
plt.plot(x, nn.predict(x), 'r')
plt.plot(x, linreg.predict(x), 'g-.')
plt.legend(['Neural Network', 'Linear Regression'])
plt.show()
