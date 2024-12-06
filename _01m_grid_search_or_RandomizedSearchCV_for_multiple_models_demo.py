"""
performing a grid search across different models (Logistic Regression and Random Forest) within the same pipeline, adjusting parameters specific to each model for optimal performance.
"""
# tune 2+ models using the same grid search
# 1. Create multiple parameter dictionaries
# 2. Specify the model within each dictionary
# 3. Put the dictionaries in a list

data_url_train = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv'
data_url_test = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_test.csv'
df = pd.read_csv(data_url_train)

import pandas as pd

cols = ['Sex', 'Name', 'Age']
X = df[cols]
y = df['Survived']
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# create Pipeline
ct = ColumnTransformer(
    [('ohe', OneHotEncoder(), ['Sex']),
     ('vectorizer', CountVectorizer(), 'Name'),
     ('imputer', SimpleImputer(), ['Age'])])

# each of these models will take a turn as the 2nd Pipeline step
clf1 = LogisticRegression(solver='liblinear', random_state=1)
clf2 = RandomForestClassifier(random_state=1)

# create the Pipeline
pipe = Pipeline([('preprocessor', ct), ('classifier', clf1)]) # ct --> clf1

# create the parameter dictionary for clf1
params1 = {}
params1['preprocessor__vectorizer__ngram_range'] = [(1, 1), (1, 2)]
params1['classifier__penalty'] = ['l1', 'l2']
params1['classifier__C'] = [0.1, 1, 10]
params1['classifier'] = [clf1]

# create the parameter dictionary for clf2
params2 = {}
params2['preprocessor__vectorizer__ngram_range'] = [(1, 1), (1, 2)]
params2['classifier__n_estimators'] = [100, 200]  # RandomForest params
params2['classifier__min_samples_leaf'] = [1, 2]  # RandomForest params
params2['classifier'] = [clf2]  # during the grid search, this will override the previous classifier when creating/running the pipeline

# create a list of parameter dictionaries
params = [params1, params2]

# search every parameter combination within each dictionary
grid = GridSearchCV(pipe, params)
grid.fit(X, y)

# what is best score found during the search?
print(grid.best_score_)

import json

# Function to recursively convert non-serializable objects to strings
def serialize_best_params(params):
    serialized_params = {}
    for key, value in params.items():
        if isinstance(value, (LogisticRegression, RandomForestClassifier)):  # Check if it's a model
            serialized_params[key] = str(value.__class__.__name__)  # Convert to class name string
        elif isinstance(value, dict):  # Recursively handle dicts if necessary
            serialized_params[key] = serialize_best_params(value)
        else:
            serialized_params[key] = value
    return serialized_params

# After the grid search
best_params_serialized = serialize_best_params(grid.best_params_)

# which combination of parameters produced the best score?
print(json.dumps(best_params_serialized, indent=4))

"""
0.8282405373171804

{
    "classifier": "LogisticRegression",
    "classifier__C": 10,
    "classifier__penalty": "l1",
    "preprocessor__vectorizer__ngram_range": [
        1,
        2
    ]
}
"""

## RandomizedSearchCV if GridSearchCV is taking too long
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# reload data
X = df['Name']
y = df['Survived']

pipe = make_pipeline(CountVectorizer(), MultinomialNB())
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

# specify parameter values to search (use a distribution for any continuous parameters)
import scipy as sp
params = {}
params['countvectorizer__min_df'] = [1, 2, 3, 4]
params['countvectorizer__lowercase'] = [True, False]
params['multinomialnb__alpha'] = sp.stats.uniform(scale=1) # distribution of factors that may be `in-between the numbers`

number_of_combinatoric_factors_to_search = len(params['countvectorizer__min_df'] ) * len(params['countvectorizer__lowercase']) # * len(params['multinomialnb__alpha'])
print(f"number_of_combinatoric_factors_to_search: {number_of_combinatoric_factors_to_search}")

# try "n_iter" random combinations of those parameter values
from sklearn.model_selection import RandomizedSearchCV
rand = RandomizedSearchCV(pipe, params, n_iter=10, cv=5, scoring='accuracy', random_state=1)
rand.fit(X, y);

# what was the best score found during the search?
print(rand.best_score_)

# which combination of parameters produced the best score?
print(rand.best_params_)

# convert results into a DataFrame
results = pd.DataFrame(rand.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]

# sort by test score
results.sort_values('rank_test_score')

"""
number_of_combinatoric_factors_to_search: 8
0.8080534806352395
{'countvectorizer__lowercase': False, 'countvectorizer__min_df': 3, 'multinomialnb__alpha': 0.1981014890848788}
"""