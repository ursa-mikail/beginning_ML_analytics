data_url_train = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv'
data_url_test = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_test.csv'

# _01k_cross_validation_and_auc_demo.py
# If use cross-validation and samples are NOT in an arbitrary order (e.g. sorted), shuffling may be required to get meaningful results.
# Use KFold or StratifiedKFold in order to shuffle!

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# Regression problem
X_reg, y_reg = load_diabetes(return_X_y=True)
reg = LinearRegression()

# Classification problem
df = pd.read_csv(data_url_train)
X_clf = df[['Pclass', 'Fare', 'SibSp']]
y_clf = df['Survived']
clf = LogisticRegression()

# KFold for regression
kf = KFold(5, shuffle=True, random_state=1)
print(cross_val_score(reg, X_reg, y_reg, cv=kf, scoring='r2'))

# StratifiedKFold for classification
skf = StratifiedKFold(5, shuffle=True, random_state=1)
print(cross_val_score(clf, X_clf, y_clf, cv=skf, scoring='accuracy'))

# multiclass_auc_demo.py
# AUC is a good evaluation metric for binary classification, especially if you have class imbalance.
# AUC can be used with multiclass problems. Supports "one-vs-one" and "one-vs-rest" strategies.

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

# Load dataset
X, y = load_wine(return_X_y=True)
X = X[:, 0:2]  # Keep only 2 features in order to make this problem harder

# Train/test split # Multiclass AUC with train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_score = clf.predict_proba(X_test)
print("Multiclass AUC (One-vs-One):", roc_auc_score(y_test, y_score, multi_class='ovo')) # use 'ovo' (One-vs-One) or 'ovr' (One-vs-Rest)

# Cross-validation # Multiclass AUC with cross-validation
print("Cross-validated Multiclass AUC (One-vs-One):", cross_val_score(clf, X, y, cv=5, scoring='roc_auc_ovo').mean())

"""
[0.43843162 0.38982359 0.52792629 0.47359827 0.5744937 ]
[0.65363128 0.7247191  0.66853933 0.68539326 0.65730337]
Multiclass AUC (One-vs-One): 0.9399801587301587
Cross-validated Multiclass AUC (One-vs-One): 0.9086960878627546
"""