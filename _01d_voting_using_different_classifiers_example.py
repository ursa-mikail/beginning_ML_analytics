# To improve classifier's accuracy: Create multiple models and ensemble them using VotingClassifier.
# P.S. VotingRegressor is available (use if it is a regressor problem, the voting will always be `soft`, i.e. averaging)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')

# Select relevant columns as features
cols = ['Pclass', 'Parch', 'SibSp', 'Fare']
X = df[cols]  # Feature set
y = df['Survived']  # Target variable

# Create individual classifiers
lr = LogisticRegression(solver='liblinear', random_state=1)
rf = RandomForestClassifier(max_features=None, random_state=1)

# Evaluate the Logistic Regression classifier using cross-validation
lr_accuracy = cross_val_score(lr, X, y).mean()
print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")

# Evaluate the Random Forest classifier using cross-validation
rf_accuracy = cross_val_score(rf, X, y).mean()
print(f"Random Forest accuracy: {rf_accuracy:.4f}")

# Create an ensemble classifier with VotingClassifier (soft voting)
vc = VotingClassifier([('clf1', lr), ('clf2', rf)], voting='soft')

# Evaluate the ensemble classifier using cross-validation
vc_accuracy = cross_val_score(vc, X, y).mean()
print(f"Voting Classifier accuracy: {vc_accuracy:.4f}")

"""
Logistic Regression accuracy: 0.6836
Random Forest accuracy: 0.6948
Voting Classifier accuracy: 0.7251
"""
