# Display the intercept & coefficients for a linear model. Stored as attributes of the model: `intercept_` and `coef_`
# Attributes end with _ if they are set during the "fit" process.

# Install or upgrade scikit-learn if needed
#!pip install --upgrade scikit-learn

# Import necessary libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# Load the diabetes dataset
dataset = load_diabetes()

# Features and target variables
X, y = dataset.data, dataset.target
features = dataset.feature_names

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Display the intercept of the model (y-intercept)
print(f"Intercept: {model.intercept_}")

# Display the coefficients of the model (one for each feature)
print(f"Coefficients: {model.coef_}")

# Display the feature names along with their corresponding coefficients
print("Feature names and their coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef}")


import numpy as np
import json

# Store the intercept and coefficients in a dictionary
model_params = {
    'intercept': model.intercept_.tolist(),
    'coefficients': model.coef_.tolist(),
    'features': features
}

# Save the model parameters to a JSON file
with open('linear_model_params.json', 'w') as f:
    json.dump(model_params, f)

# Load the model parameters from the JSON file
with open('linear_model_params.json', 'r') as f:
    loaded_params = json.load(f)

# Extract the intercept and coefficients
intercept = np.array(loaded_params['intercept'])
coefficients = np.array(loaded_params['coefficients'])
features = loaded_params['features']

# Display the loaded intercept and coefficients
print(f"Loaded Intercept: {intercept}")
print("Loaded Coefficients:")
for feature, coef in zip(features, coefficients):
    print(f"{feature}: {coef}")

# Verify if the loaded coefficients and intercept can be used to make predictions
# Create a new linear regression model with the loaded parameters
loaded_model = LinearRegression()
loaded_model.intercept_ = intercept
loaded_model.coef_ = coefficients

# Check if predictions are the same
original_predictions = model.predict(X)
loaded_predictions = loaded_model.predict(X)
print(f"Predictions are the same: {np.allclose(original_predictions, loaded_predictions)}")

"""
Intercept: 152.13348416289597
Coefficients: [ -10.0098663  -239.81564367  519.84592005  324.3846455  -792.17563855
  476.73902101  101.04326794  177.06323767  751.27369956   67.62669218]
Feature names and their coefficients:
age: -10.009866299810684
sex: -239.81564367242237
bmi: 519.84592005446
bp: 324.38464550232356
s1: -792.1756385522286
s2: 476.7390210052569
s3: 101.04326793803338
s4: 177.06323767134643
s5: 751.2736995571034
s6: 67.62669218370515

Loaded Intercept: 152.13348416289597
Loaded Coefficients:
age: -10.009866299810684
sex: -239.81564367242237
bmi: 519.84592005446
bp: 324.38464550232356
s1: -792.1756385522286
s2: 476.7390210052569
s3: 101.04326793803338
s4: 177.06323767134643
s5: 751.2736995571034
s6: 67.62669218370515
Predictions are the same: True
"""
