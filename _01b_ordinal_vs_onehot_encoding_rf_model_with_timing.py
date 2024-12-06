import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Load the dataset
df = pd.read_csv('https://www.openml.org/data/get_csv/1595261/adult-census.csv')

# List of categorical columns
categorical_cols = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex']

# Features (categorical columns) and target variable
X = df[categorical_cols]
y = df['class']

# Using OneHotEncoder (creates many columns for each unique category)
ohe = OneHotEncoder(sparse_output=False)  # sparse_output=False ensures we get a dense matrix
ohe.fit_transform(X).shape  # Output will be (48842, 60) for this dataset
# OneHotEncoder creates 60 columns for the 7 categorical features.

# Using OrdinalEncoder (one column per feature, encoding categories with integers)
oe = OrdinalEncoder()
oe.fit_transform(X).shape  # Output will be (48842, 7) since each feature is encoded as a single column
# OrdinalEncoder creates 7 columns (one for each feature).

# Random Forest Classifier (tree-based model)
rf = RandomForestClassifier(random_state=1, n_jobs=-1)

# Pipeline with OneHotEncoder
ohe_pipe = make_pipeline(ohe, rf)
# Evaluate the pipeline using cross-validation
print(f"Using OneHotEncoder: {cross_val_score(ohe_pipe, X, y).mean()}")
%time cross_val_score(ohe_pipe, X, y).mean() # cross-validation accuracy score 
# This will take longer due to the high dimensionality created by OneHotEncoder.

# Pipeline with OrdinalEncoder
oe_pipe = make_pipeline(oe, rf)
# Evaluate the pipeline using cross-validation
print(f"Using OrdinalEncoder: {cross_val_score(oe_pipe, X, y).mean()}")
%time cross_val_score(oe_pipe, X, y).mean() # cross-validation accuracy score 
# This will be faster since OrdinalEncoder creates fewer columns (7 instead of 60).


"""
# With a tree-based model, try OrdinalEncoder instead of OneHotEncoder even for nominal (unordered) features.
# Accuracy will often be similar, but OrdinalEncoder will be much faster.

Using OneHotEncoder: 0.8262561170407418
CPU times: user 29.5 s, sys: 162 ms, total: 29.7 s
Wall time: 17.7 s

Using OrdinalEncoder: 0.8256623624061437
CPU times: user 17.3 s, sys: 140 ms, total: 17.4 s
Wall time: 11.1 s

"""