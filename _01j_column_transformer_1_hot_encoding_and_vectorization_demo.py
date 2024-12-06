data_url_train = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv'
data_url_test = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_test.csv'

# vectorize 2 text columns in a ColumnTransformer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer

# Load dataset
df = pd.read_csv(data_url_train)
X = df[['Name', 'Cabin']].dropna()

# Create ColumnTransformer
vect = CountVectorizer()
ct = make_column_transformer((vect, 'Name'), (vect, 'Cabin'))

# Apply transformations
transformed = ct.fit_transform(X)
print(transformed)

"""
  (0, 105)	1
  (0, 336)	1
  (0, 242)	1
  (0, 66)	1
  (0, 157)	1
  (0, 67)	1
  (0, 432)	1
  (0, 572)	1
  (1, 336)	1
  (1, 173)	1
  (1, 235)	1
  :	:
  (203, 46)	1
  (203, 248)	1
  (203, 221)	1
  (203, 547)	1
"""
import numpy as np

# Selecting relevant features
df = df[['Survived', 'Age', 'Fare', 'Pclass', 'Sex', 'Name']]
cols = ['Sex', 'Name']
X = df[cols]
y = df['Survived']

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()
vect = CountVectorizer()
ct = make_column_transformer((ohe, ['Sex']), (vect, 'Name'))

# Apply transformations
transformed = ct.fit_transform(X)
print(transformed)

"""
  (0, 1)	1.0
  (0, 179)	1.0
  (0, 1014)	1.0
  (0, 1098)	1.0
  (0, 582)	1.0
  (1, 0)	1.0
  :	:
  (2, 985)	1.0
  (2, 788)	1.0
  (3, 0)	1.0
  (3, 1015)	1.0
  (3, 492)	1.0
  (3, 678)	1.0
  :	:
  (889, 745)	1.0
  (889, 135)	1.0
  (889, 639)	1.0
  (890, 1)	1.0
  (890, 1014)	1.0
  (890, 1111)	1.0
  (890, 349)	1.0
"""
