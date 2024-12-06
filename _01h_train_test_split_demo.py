import pandas as pd
from sklearn.model_selection import train_test_split

# Create simulated dataset
df = pd.DataFrame({'feature': list(range(8)), 'target': ['not fraud']*6 + ['fraud']*2})
X = df[['feature']]
y = df['target']

# Split without stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
print("Without Stratification:")
print("y_train:", y_train)
print("y_test:", y_test)

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
print("With Stratification:")
print("y_train:", y_train)
print("y_test:", y_test)

"""
Without Stratification:
y_train: 3    not fraud
0    not fraud
5    not fraud
4    not fraud
Name: target, dtype: object
y_test: 6        fraud
2    not fraud
1    not fraud
7        fraud
Name: target, dtype: object
With Stratification:
y_train: 1    not fraud
7        fraud
2    not fraud
4    not fraud
Name: target, dtype: object
y_test: 3    not fraud
6        fraud
0    not fraud
5    not fraud
Name: target, dtype: object
"""
