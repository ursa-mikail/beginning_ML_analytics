import pandas as pd
cols = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')
X = df[cols]
y = df['Survived']
df_new = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_test.csv', nrows=10)
X_new = df_new[cols]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# set up preprocessing for numeric columns
imp_median = SimpleImputer(strategy='median', add_indicator=True) # True: impute `missing`
scaler = StandardScaler() # variance in the same order, centered around 0

# set up preprocessing for categorical columns
imp_constant = SimpleImputer(strategy='constant') # treat missing as separate category
ohe = OneHotEncoder(handle_unknown='ignore')

# select columns by data type
num_cols = make_column_selector(dtype_include='number') # select of numeric columns
cat_cols = make_column_selector(dtype_exclude='number') # select of non-numeric columns

# do all preprocessing (map impute transformer to column selected)
preprocessor = make_column_transformer(
    (make_pipeline(imp_median, scaler), num_cols),
    (make_pipeline(imp_constant, ohe), cat_cols))

# create a pipeline
pipe = make_pipeline(preprocessor, LogisticRegression()) # 1. column transformer for pre-processing; 2. LogisticRegression for classification
# cross-validate the pipeline
cross_val_score(pipe, X, y).mean() # 0.8035904839620865

# fit the pipeline and make predictions
pipe.fit(X, y)
pipe.predict(X_new)

"""
array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
"""

# # to operate on part of a Pipeline (instead of the whole thing): Slice it using Python's slicing notation.
cols = ['Sex', 'Name', 'Age']
X = df[cols]
y = df['Survived']
from sklearn import set_config
set_config(display='diagram')
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
ct = ColumnTransformer(
    [('ohe', OneHotEncoder(), ['Sex']),
     ('vectorizer', CountVectorizer(), 'Name'),
     ('imputer', SimpleImputer(), ['Age'])])
fs = SelectPercentile(chi2, percentile=50)
clf = LogisticRegression(solver='liblinear', random_state=1)
# create Pipeline
pipe = Pipeline([('preprocessor', ct), ('feature selector', fs), ('classifier', clf)])

from sklearn import set_config
set_config(display='diagram')

from IPython.display import display
display(pipe)
# pipe

# access step 0 (preprocessor)
pipe[0].fit_transform(X)

# access steps 0 and 1 (preprocessor and feature selector)
pipe[0:2].fit_transform(X, y)

# access step 1 (feature selector), which of the 756 features are selected
pipe[1].get_support()

"""
array([ True,  True,  True, ...,  True, False,  True])
"""
