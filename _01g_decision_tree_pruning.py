# Pruning of decision trees to avoid overfitting.
# Uses cost-complexity pruning.
# Increase "ccp_alpha" to increase pruning (default value is 0).

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')

# Convert 'Sex' feature to numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Define features and target variable
features = ['Pclass', 'Fare', 'Sex', 'Parch']
X = df[features]
y = df['Survived']

# Initialize and train a default DecisionTreeClassifier
# Default tree has 331 nodes
dt_default = DecisionTreeClassifier(random_state=0)
dt_default.fit(X, y)
default_node_count = dt_default.tree_.node_count
print(f"Default tree node count: {default_node_count}")

# Cross-validated accuracy of the default tree
default_cv_accuracy = cross_val_score(dt_default, X, y, cv=5, scoring='accuracy').mean()
print(f"Default tree cross-validated accuracy: {default_cv_accuracy:.4f}")

# Initialize and train a pruned DecisionTreeClassifier with ccp_alpha=0.001
# Pruned tree has fewer nodes
dt_pruned = DecisionTreeClassifier(ccp_alpha=0.001, random_state=0)
dt_pruned.fit(X, y)
pruned_node_count = dt_pruned.tree_.node_count
print(f"Pruned tree node count: {pruned_node_count}")

# Cross-validated accuracy of the pruned tree
pruned_cv_accuracy = cross_val_score(dt_pruned, X, y, cv=5, scoring='accuracy').mean()
print(f"Pruned tree cross-validated accuracy: {pruned_cv_accuracy:.4f}")

# Compare the number of nodes and accuracy before and after pruning
print(f"Node count reduction: {default_node_count - pruned_node_count}")
print(f"Accuracy improvement: {pruned_cv_accuracy - default_cv_accuracy:.4f}")

"""
Default tree node count: 331
Default tree cross-validated accuracy: 0.8036
Pruned tree node count: 121
Pruned tree cross-validated accuracy: 0.8081
Node count reduction: 210
Accuracy improvement: 0.0045
"""