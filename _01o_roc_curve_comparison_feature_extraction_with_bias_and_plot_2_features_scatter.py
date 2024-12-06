# reflects the main focus of the code, which is comparing multiple ROC curves in a single plot for different classifiers.

data_url_train = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv'
data_url_test = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_test.csv'
df = pd.read_csv(data_url_train)

cols = ['Pclass', 'Fare', 'SibSp']
X = df[cols]
y = df['Survived']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lr.fit(X_train, y_train);
dt.fit(X_train, y_train);
rf.fit(X_train, y_train);

# Compare multiple ROC curves in a single plot
disp = RocCurveDisplay.from_estimator(lr, X_test, y_test)
RocCurveDisplay.from_estimator(dt, X_test, y_test, ax=disp.ax_);
RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=disp.ax_);


# feature_extraction_with_bias.py.
# reflects the process of feature extraction, including adding a bias term to the feature matrix, as seen in the code.
# 
import numpy as np
import pandas as pd

data_len = 10
data_ex = 1
# Example training data
train_data = pd.DataFrame({
    'feature1': np.random.rand(data_len),
    'feature2': np.random.rand(data_len),
    # Add more features if needed
    'target': np.random.randint(0, 2, size=data_len)  # Binary classification target
})

# Example test data
test_data = pd.DataFrame({
    'feature1': np.random.rand(data_len + data_ex),
    'feature2': np.random.rand(data_len + data_ex),
    # Add more features if needed
    'target': np.random.randint(0, 2, size=data_len + data_ex)  # Binary classification target
})

# Extracting features from training and test data
X = train_data.iloc[:, :-1].values
test_data_values = test_data.values
X_test = test_data_values[:, :-1]

# Printing shapes
print("Shape of train_data :", train_data.shape)
print("Shape of X_train :", X.shape)
print("Shape of X_test :", X_test.shape)

# Adding bias term to the feature matrix X
X_with_bias = np.vstack((np.ones((X.shape[0], )), X.T)).T
print("Shape of X_train with bias term:", X_with_bias.shape)

print("X_test :", X_test)
print("---> :", (np.ones((X.shape[0], ))))
print("-> :", X.shape)
print("-> :", X.T.shape)
print("--- :", np.vstack((np.ones((X.shape[0], )), X.T)).shape)
print("--- :", np.vstack((np.ones((X.shape[0], )), X.T)))

"""
Shape of train_data : (10, 3)
Shape of X_train : (10, 2)
Shape of X_test : (11, 2)
Shape of X_train with bias term: (10, 3)
X_test : [[0.64688802 0.46351264]
 [0.38133486 0.15552826]
 [0.89798161 0.15923454]
 [0.78346524 0.45939647]
 [0.71571186 0.52619116]
 [0.58640074 0.07509117]
 [0.42033016 0.62643914]
 [0.62405144 0.50989272]
 [0.43686953 0.62282612]
 [0.67938279 0.06358749]
 [0.21772462 0.09731429]]
---> : [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
-> : (10, 2)
-> : (2, 10)
--- : (3, 10)
--- : [[1.         1.         1.         1.         1.         1.
  1.         1.         1.         1.        ]
 [0.09975067 0.58162301 0.04148773 0.63034408 0.28275801 0.50095103
  0.17582332 0.1530281  0.5442813  0.36859954]
 [0.13997064 0.7653313  0.01417497 0.55478705 0.847926   0.77213692
  0.99256517 0.90813523 0.59061146 0.12015432]]
"""

import matplotlib.pyplot as plt

# Assuming X_test has 2 features
if X_test.shape[1] == 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c='b', alpha=0.7)
    plt.title('Scatter plot of X_test')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()
elif X_test.shape[1] == 3:  # If X_test has 3 features, plot in 3D
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c='b', marker='o', alpha=0.7)
    ax.set_title('3D Scatter plot of X_test')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.show()
else:
    print("Number of features in X_test > 3. Use dimensionality reduction techniques for visualization.")

