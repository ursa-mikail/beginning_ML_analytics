# _01l_function_transformer_and_select_percentile_demo.py
# Select an existing function (or write your own)
# Convert it into a transformer using FunctionTransformer
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

data_url_train = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv'
data_url_test = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_test.csv'
df = pd.read_csv(data_url_train)

# Custom function to extract the first letter. Convert custom function into a transformer.
def first_letter(df):
    return df.apply(lambda x: x.str.slice(0, 1))

# Dataset
X = pd.DataFrame({'Fare':[200, 300, 50, 900],
                  'Code':['X12', 'Y20', 'Z7', np.nan],
                  'Deck':['A101', 'C102', 'A200', 'C300']})
# Create FunctionTransformers. Convert existing function into a transformer:
clip_values = FunctionTransformer(np.clip, kw_args={'a_min': 100, 'a_max': 600})
get_first_letter = FunctionTransformer(first_letter)

# ColumnTransformer. Include them in a ColumnTransformer:
ct = make_column_transformer(
    (clip_values, ['Fare']),
    (get_first_letter, ['Code', 'Deck']))

# Apply transformations
print(ct.fit_transform(X))

# select_percentile_demo.py
# Use SelectPercentile to keep the highest scoring features
# Add feature selection after preprocessing but before model building
# P.S. Make sure to tune the percentile value.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv(data_url_train)
X = df['Name']
y = df['Survived']

# Pipeline without feature selection
vect = CountVectorizer()
clf = LogisticRegression()
pipe = make_pipeline(vect, clf)
print("Pipeline without feature selection:", cross_val_score(pipe, X, y, scoring='accuracy').mean())

# Pipeline with feature selection
selection = SelectPercentile(chi2, percentile=50) # keep 50% of features with the best chi-squared scores
pipe = make_pipeline(vect, selection, clf) # recommended: transformer, feature selection, regressor or classifier
print("Pipeline with feature selection:", cross_val_score(pipe, X, y, scoring='accuracy').mean())

"""
[[200 'X' 'A']
 [300 'Y' 'C']
 [100 'Z' 'A']
 [600 nan 'C']]
Pipeline without feature selection: 0.7957190383528967
Pipeline with feature selection: 0.8159060950348378
"""