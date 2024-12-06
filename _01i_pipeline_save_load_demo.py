import pandas as pd

data_url_train = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv'
data_url_test = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_test.csv'
 
df = pd.read_csv(data_url_train) 

number_of_rows_to_read = len(df) # Count the number of rows in the dataset

header = df.columns.tolist()    # Get the header (column names)
data = df.values.tolist()       # Get the data as a list of lists

print("Header:", header)
print("Number of rows:", number_of_rows_to_read)
print("Data:", len(data))

# Create a new DataFrame with the same columns
table = df[header]

# Display the first few rows of the new DataFrame to confirm
print(table.head())

"""
Header: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
Number of rows: 891
Data: 891
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  
"""

# Warning: You must load it into an identical environment, and only load objects you trust 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Load datasets
cols = ['Embarked', 'Sex']
df = pd.read_csv(data_url_train, nrows=10)
X = df[cols]
y = df['Survived']
df_new = pd.read_csv(data_url_test, nrows=10)
X_new = df_new[cols]

# Create and fit pipeline
ohe = OneHotEncoder()
logreg = LogisticRegression(solver='liblinear', random_state=1)
pipe = make_pipeline(ohe, logreg)
pipe.fit(X, y)

# Save pipeline
file_to_save_to = './sample_data/pipe.joblib'
joblib.dump(pipe, file_to_save_to)

# Load pipeline
same_pipe = joblib.load(file_to_save_to)

# Make predictions
predictions = same_pipe.predict(X_new)
print(predictions)

# [0 1 0 0 1 0 1 0 1 0]

from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv(data_url_train, usecols=['Embarked', 'Survived']).dropna()
X = df[['Embarked']]
y = df['Survived']

# Create and fit pipeline
pipe = Pipeline([('ohe', OneHotEncoder()), ('clf', LogisticRegression())])
pipe.fit(X, y)

# Display model coefficients. 
print(f"4 ways to display the model coefficients:")
print(pipe.named_steps.clf.coef_)
print(pipe.named_steps['clf'].coef_)
print(pipe['clf'].coef_)
step = 1
print(pipe[step].coef_)

"""
4 ways to display the model coefficients:
[[ 0.43735139 -0.20614895 -0.44031791]]
[[ 0.43735139 -0.20614895 -0.44031791]]
[[ 0.43735139 -0.20614895 -0.44031791]]
[[ 0.43735139 -0.20614895 -0.44031791]]
"""

# pipeline_evaluation.py
# pipeline_steps_demo.py: ways to examine the steps of a Pipeline
# (Prefer method 1 since you can autocomplete the step & parameter names... though method 4 is short)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Load the preprocessed data (assuming loading_preprocessing.py has been run)
df = pd.read_csv(data_url_train)
df = df[['Survived', 'Age', 'Fare', 'Pclass', 'Sex', 'Name']]
cols = ['Sex', 'Name']
X = df[cols]
y = df['Survived']

# Define the column transformer and pipeline
ohe = OneHotEncoder()
vect = CountVectorizer()
ct = make_column_transformer((ohe, ['Sex']), (vect, 'Name'))

clf = LogisticRegression(solver='liblinear', random_state=1)
pipe = make_pipeline(ct, clf)

# Cross-validate the pipeline
ans = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
print(ans) # 0.8024543343167408