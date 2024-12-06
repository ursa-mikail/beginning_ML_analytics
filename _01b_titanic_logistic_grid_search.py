# Import necessary libraries
import pandas as pd

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')

# Select relevant columns for the features (X) and target variable (y)
cols = ['Sex', 'Name', 'Age']
X = df[cols]
y = df['Survived']

# Configure scikit-learn to display pipeline diagrams
from sklearn import set_config
set_config(display='diagram')

# Import necessary preprocessing and modeling tools
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Define a ColumnTransformer for preprocessing:
# - OneHotEncoder for the 'Sex' feature (encoding categorical data)
# - CountVectorizer for the 'Name' feature (turning text into a bag-of-words representation)
# - SimpleImputer for the 'Age' feature (handling missing values by imputing)
ct = ColumnTransformer(
    [('ohe', OneHotEncoder(), ['Sex']),
     ('vectorizer', CountVectorizer(), 'Name'),
     ('imputer', SimpleImputer(), ['Age'])])

# Initialize a Logistic Regression model (used for classification)
clf = LogisticRegression(solver='liblinear', random_state=1)

# Define the pipeline to include both preprocessing and the classifier
pipe = Pipeline([('preprocessor', ct), ('classifier', clf)])

# Set up the hyperparameter grid for tuning:
# - Tuning OneHotEncoder's 'drop' parameter
# - Tuning CountVectorizer's 'min_df' and 'ngram_range' parameters
# - Tuning Logistic Regression's 'C' and 'penalty' parameters
params = {}
params['preprocessor__ohe__drop'] = [None, 'first']
params['preprocessor__vectorizer__min_df'] = [1, 2, 3]
params['preprocessor__vectorizer__ngram_range'] = [(1, 1), (1, 2)]
params['classifier__C'] = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
params['classifier__penalty'] = ['l1', 'l2']

# Perform GridSearchCV to tune the hyperparameters and fit the model
grid = GridSearchCV(pipe, params)
%time grid.fit(X, y)  # Measure the time taken to perform grid search and model fitting

# Perform GridSearchCV with parallel processing (using all available CPUs) for faster execution
grid = GridSearchCV(pipe, params, n_jobs=-1)
%time grid.fit(X, y)  # Measure the time taken with parallel processing

""" # grid search to run faster? Set n_jobs=-1 to use parallel processing with all CPUs

CPU times: user 44.4 s, sys: 190 ms, total: 44.6 s
Wall time: 53.4 s

CPU times: user 2.05 s, sys: 74.1 ms, total: 2.13 s
Wall time: 26.9 s
"""
