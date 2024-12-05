#Create Random Data Points
import numpy as np
from sklearn.utils import shuffle
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs

X, Y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=5, random_state=11)

def next_batch(X, Y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], Y[i:i + batch_size])

print(X.shape)
X = np.hstack((np.ones((X.shape[0], 1)), X)) # stack on the left side (# Adding column of 1's on the left). same as: X = np.c_[np.ones((X.shape[0])), X]
#print(X)
print(X.shape)
X = np.vstack((np.ones((1, X.shape[1])), X)) # stack on top
#print(X)

print(X.shape)

W = np.random.uniform(size=(X.shape[1],))
print(W.shape)

import matplotlib.pyplot as plt

# Scatter plot of the data points
def plot_data(X, Y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[Y == 0][:, 1], X[Y == 0][:, 2], color='red', label='Class 0')
    plt.scatter(X[Y == 1][:, 1], X[Y == 1][:, 2], color='blue', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Scatter Plot of the Data Points')
    plt.show()

plot_data(X[1:], Y)  # Skipping the first row of X which is the additional row of ones


import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression cost function
def compute_cost(X, Y, W):
    m = len(Y)
    h = sigmoid(X.dot(W))
    epsilon = 1e-5  # To avoid log(0) error
    cost = -(1/m) * np.sum(Y * np.log(h + epsilon) + (1 - Y) * np.log(1 - h + epsilon))
    return cost

# Gradient descent for logistic regression
def gradient_descent(X, Y, W, alpha, num_iters):
    m = len(Y)
    cost_history = []

    for i in range(num_iters):
        gradient = (1/m) * X.T.dot(sigmoid(X.dot(W)) - Y)
        W -= alpha * gradient
        cost = compute_cost(X, Y, W)
        cost_history.append(cost)

    return W, cost_history

# Initializing weights, learning rate and number of iterations
W = np.random.uniform(size=(X.shape[1],))
alpha = 0.01
num_iters = 1000

# Running gradient descent
W, cost_history = gradient_descent(X[1:], Y, W, alpha, num_iters)

# Plotting the cost history
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History over Iterations')
plt.show()


def plot_decision_boundary(X, Y, W):
    plot_data(X[1:], Y)  # Skipping the first row of X which is the additional row of ones

    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]
    probs = sigmoid(grid.dot(W)).reshape(xx.shape)
    
    plt.contourf(xx, yy, probs, alpha=0.8, levels=[0, 0.5, 1], colors=['red', 'blue'], linestyles=['--'])
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(X, Y, W)

"""
(300, 2)
(300, 3)
(301, 3)
(3,)
"""