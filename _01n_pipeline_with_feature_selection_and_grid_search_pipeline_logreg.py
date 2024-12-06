# pipeline_with_feature_selection.py
# reflects that the code builds a pipeline involving feature selection using SelectPercentile along with preprocessing steps and model fitting.
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

X = df[['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']]
y = df['Survived']
imp_constant = SimpleImputer(strategy='constant')
ohe = OneHotEncoder()
imp_ohe = make_pipeline(imp_constant, ohe)
vect = CountVectorizer()
imp = SimpleImputer()
# pipeline step 1
ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch']))
# pipeline step 2
selection = SelectPercentile(chi2, percentile=50)
# pipeline step 3
logreg = LogisticRegression(solver='liblinear')
# display estimators as diagrams
from sklearn import set_config
set_config(display='diagram')
pipe = make_pipeline(ct, selection, logreg)

# export the diagram to a file
from sklearn.utils import estimator_html_repr
with open('pipeline.html', 'w') as f:
    f.write(estimator_html_repr(pipe))

pipe

# grid_search_pipeline_logreg.py.
# reflects the use of GridSearchCV with a LogisticRegression model in a pipeline, specifically tuned with hyperparameters for logistic regression.
X = df[['Pclass', 'Sex', 'Name']]
y = df['Survived']
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

ohe = OneHotEncoder()
vect = CountVectorizer()
clf = LogisticRegression(solver='liblinear', random_state=1)
ct = make_column_transformer((ohe, ['Sex']), (vect, 'Name'), remainder='passthrough')
pipe = Pipeline([('preprocessor', ct), ('model', clf)])
# specify parameter values to search
params = {}
params['model__C'] = [0.1, 1, 10]
params['model__penalty'] = ['l1', 'l2']
# try all possible combinations of those parameter values
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y);
# convert results into a DataFrame
results = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]
# sort by test score
results.sort_values('rank_test_score')

"""
params	mean_test_score	rank_test_score
4	{'model__C': 10, 'model__penalty': 'l1'}	0.821537	1
2	{'model__C': 1, 'model__penalty': 'l1'}	0.820394	2
5	{'model__C': 10, 'model__penalty': 'l2'}	0.817055	3
3	{'model__C': 1, 'model__penalty': 'l2'}	0.812573	4
1	{'model__C': 0.1, 'model__penalty': 'l2'}	0.791225	5
0	{'model__C': 0.1, 'model__penalty': 'l1'}	0.788984	6
"""
