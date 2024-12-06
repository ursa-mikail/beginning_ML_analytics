import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, mean_squared_error



# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')

# Select columns
all_cols = df.columns.tolist()
print("All columns:", all_cols)

# Randomly choose a column for y
y_col = random.choice(all_cols)
y = df[y_col]
print("Chosen target column (y):", y_col)

# Select N-1 columns for X (excluding the chosen y column)
X_cols = [col for col in all_cols if col != y_col]
X = df[X_cols]
print("Feature columns (X):", X_cols)

# Load new data for prediction
df_new = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_test.csv', nrows=10)
X_new = df_new[[col for col in X_cols if col in df_new.columns]]

# Handle text columns
if 'Name' in X_cols:
    X['Name_Length'] = df['Name'].apply(len)
    df_new['Name_Length'] = df_new['Name'].apply(len)
    X_new['Name_Length'] = df_new['Name_Length']

# Set up preprocessing for numeric columns
imp_median = SimpleImputer(strategy='median', add_indicator=True)
scaler = StandardScaler()

# Set up preprocessing for categorical columns
imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
ohe = OneHotEncoder(handle_unknown='ignore')

# Select columns by data type
num_cols = make_column_selector(dtype_include=['number'])
cat_cols = make_column_selector(dtype_include=['object'])

# Do all preprocessing
preprocessor = make_column_transformer(
    (make_pipeline(imp_median, scaler), num_cols),
    (make_pipeline(imp_constant, ohe), cat_cols)
)

# Impute missing values in the target column if necessary
if y.isnull().any():
    if y.dtype == 'object':
        y = SimpleImputer(strategy='most_frequent').fit_transform(y.values.reshape(-1, 1)).ravel()
    else:
        y = SimpleImputer(strategy='median').fit_transform(y.values.reshape(-1, 1)).ravel()

# Determine if the problem is classification or regression based on y
if y.dtype == 'object' or len(np.unique(y)) < 20:  # Assume classification if target is categorical or has fewer than 20 unique values
    model = RandomForestClassifier(n_estimators=100)
    problem_type = 'classification'
else:
    model = RandomForestRegressor(n_estimators=100)
    problem_type = 'regression'

# Create a pipeline
pipe = make_pipeline(preprocessor, model)

# Cross-validate the pipeline
cv_score = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error' if problem_type == 'regression' else 'accuracy').mean()
print("Cross-validation score:", cv_score)

# Split the data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipe.fit(X_train, y_train)

# Make predictions
y_pred = pipe.predict(X_test)

if problem_type == 'classification':
    # Evaluate the classification model
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot ROC curve
    if len(np.unique(y)) == 2:  # Ensure y is binary for ROC curve
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
else:
    # Evaluate the regression model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Plot true vs predicted values
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.show()

"""
All columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
Chosen target column (y): Sex
Feature columns (X): ['PassengerId', 'Survived', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

Cross-validation score: 0.8136965664427844
Classification Report:
               precision    recall  f1-score   support

      female       0.82      0.74      0.78        69
        male       0.85      0.90      0.87       110

    accuracy                           0.84       179
   macro avg       0.83      0.82      0.83       179
weighted avg       0.84      0.84      0.84       179

Confusion Matrix:
 [[51 18]
 [11 99]]
"""
