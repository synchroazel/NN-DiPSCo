import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class Perceptron:

    def activaction_function(self, z):  # step function

        if z >= 0:
            return 1
        else:
            return 0

    def predict(self, X):

        pred = []
        for i in range(X.shape[0]):
            z = np.dot(self.w, X[i, :]) + self.b
            pred.append(self.activaction_function(z))
        return np.array(pred)

    def train(self, X, y, lr=.01, max_epochs=1000):

        self.w = np.random.rand(X.shape[1])
        self.b = np.random.rand()

        self.cost = []
        epoch = 0
        converged = False
        while not converged and epoch <= max_epochs:

            # epoch
            for i in range(len(y)):

                z = np.dot(self.w, X[i, :]) + self.b
                a = self.activaction_function(z)

                error = y[i] - a

                if error != 0:
                    self.w += lr * error * X[i, :]
                    self.b += lr * error

            epoch += 1
            loss = sum(abs(y - self.predict(X)))
            self.cost.append(loss)
            converged = loss == 0

            print('Epoch: %d\tLoss: %1.1f' % (epoch, loss))

            if converged:           print('\nIt converged!')
            if epoch == max_epochs: print('\nEpochs limit reached!')


# %% TEST

iris = datasets.load_iris()
X = iris.data[:100, 0:2]
y = iris.target[:100]

# shuffle dataset
idx = np.arange(y.shape[0])
np.random.shuffle(idx)
X = X[idx, :]
y = y[idx]

perc = Perceptron()
perc.train(X, y, lr=.001, max_epochs=150)

pred = perc.predict(X)
accuracy = 100 * sum(y == pred) / float(len(y))
print('\nAccuracy: %1.1f' % accuracy)

plt.plot(perc.cost)
