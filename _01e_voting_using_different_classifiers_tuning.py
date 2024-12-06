# To improve the accuracy of VotingClassifier: Try tuning the 'voting' and 'weights' parameters to change how predictions are combined!
# P.S. If using VotingRegressor, tune the 'weights' parameter as voting is not involved.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')

# Select relevant columns as features
cols = ['Pclass', 'Parch', 'SibSp', 'Fare']
X = df[cols]  # Feature set
y = df['Survived']  # Target variable

# Create individual classifiers
lr = LogisticRegression(solver='liblinear', random_state=1)
rf = RandomForestClassifier(max_features=None, random_state=1)
nb = MultinomialNB()

# Create an ensemble classifier with VotingClassifier (default hard voting)
vc = VotingClassifier([('clf1', lr), ('clf2', rf), ('clf3', nb)])

# Evaluate the initial ensemble classifier using cross-validation
vc_accuracy = cross_val_score(vc, X, y).mean()
print(f"Initial Voting Classifier accuracy: {vc_accuracy:.4f}")

# Define parameters to search for tuning 'voting' and 'weights'
params = {'voting': ['hard', 'soft'],
          'weights': [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]}

# Use GridSearchCV to find the best parameters for the ensemble model
grid = GridSearchCV(vc, params)
grid.fit(X, y)

# Display the best parameters found by GridSearchCV
print(f"Best parameters found: {grid.best_params_}")

# Display the improved accuracy score after parameter tuning
print(f"Improved Voting Classifier accuracy: {grid.best_score_:.4f}")


"""
Initial Voting Classifier accuracy: 0.6971
Best parameters found: {'voting': 'soft', 'weights': (1, 2, 1)}
Improved Voting Classifier accuracy: 0.7263
"""
