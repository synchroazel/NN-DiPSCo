# -*- coding: utf-8 -*-
"""
3 neuron NN
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class NN:
    def __init__(self):
        self.w1 = np.random.rand()
        self.w2 = np.random.rand()

        self.b1 = 0
        self.b2 = 0

    def act(self, z):
        return 1 / (1 + np.exp(-z))  # sigmoid

    def fit(self, x, y, lr=.01, max_e=1000):

        self.loss = []

        for e in range(max_e):
            loss_e = []
            for i in range(len(y)):
                """ Forward """

                z1 = x[i] * self.w1 + self.b1
                a1 = self.act(z1)

                z2 = a1 * self.w2 + self.b2
                a2 = self.act(z2)

                yhat = a2  # output

                """ Backward """
                loss_e.append((yhat - y[i]) ** 2)

                dL_yhat = 2 * (yhat - y[i])
                dyhat_z2 = self.act(z2) * (1 - self.act(z2))
                dz2_w2 = a1

                dL_w2 = dL_yhat * dyhat_z2 * dz2_w2

                dz2_a1 = self.w2
                da1_z1 = self.act(z1) * (1 - self.act(z1))
                dz1_w1 = x[i]

                dL_w1 = dL_yhat * dyhat_z2 * dz2_a1 * da1_z1 * dz1_w1

                """ Update parameters """

                self.w2 -= lr * dL_w2
                self.w1 -= lr * dL_w1

            self.loss.append(np.mean(loss_e))

    def predict(self, x):
        z1 = x * self.w1 + self.b1
        a1 = self.act(z1)

        z2 = a1 * self.w2 + self.b2
        return self.act(z2)


iris = datasets.load_iris()
X = iris.data[:100, 1]
y = iris.target[:100]

nn = NN()

nn.fit(X, y, lr=.01, max_e=5000)

plt.plot(nn.loss)

s = 89
nn.predict(X[s])

y[s]
