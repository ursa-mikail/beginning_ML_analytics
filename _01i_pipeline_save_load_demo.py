import pandas as pd

data_url = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv'
 
df = pd.read_csv(data_url) 

# Count the number of rows in the dataset
number_of_rows_to_read = len(df)

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
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv', nrows=10)
X = df[cols]
y = df['Survived']
df_new = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_test.csv', nrows=10)
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