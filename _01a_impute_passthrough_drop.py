import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

impute = SimpleImputer()  # Create a SimpleImputer instance to fill missing values

# Create a DataFrame with some missing values
X = pd.DataFrame({
    'A': [1, 2, np.nan],
    'B': [10, 20, 30],
    'C': [100, 200, 300],
    'D': [1000, 2000, 3000],
    'E': [10000, 20000, 30000]
})

# Display the original DataFrame
print("Original DataFrame:")
print(X)

# Impute missing values in column A, pass through columns B and C, and drop the remaining columns
ct = make_column_transformer(
    (impute, ['A']),             # Apply imputer to column A
    ('passthrough', ['B', 'C']), # Include columns B and C without any transformation
    remainder='drop'             # Drop all other columns (D and E)
)

# Transform the DataFrame using the column transformer
transformed_X = ct.fit_transform(X)

# Display the transformed DataFrame
print("\nTransformed DataFrame (impute A, passthrough B & C, drop others):")
print(pd.DataFrame(transformed_X, columns=['A', 'B', 'C']))

# Impute missing values in column A, drop columns D and E, and passthrough the remaining columns (B and C)
ct = make_column_transformer(
    (impute, ['A']),           # Apply imputer to column A
    ('drop', ['D', 'E']),      # Drop columns D and E
    remainder='passthrough'    # Include all other columns (B and C) without any transformation
)

# Transform the DataFrame using the column transformer
transformed_X = ct.fit_transform(X)

# Display the transformed DataFrame
print("\nTransformed DataFrame (impute A, drop D & E, passthrough others):")
print(pd.DataFrame(transformed_X, columns=['A', 'B', 'C']))


"""
Original DataFrame:
     A   B    C     D      E
0  1.0  10  100  1000  10000
1  2.0  20  200  2000  20000
2  NaN  30  300  3000  30000

Transformed DataFrame (impute A, passthrough B & C, drop others):
     A     B      C
0  1.0  10.0  100.0
1  2.0  20.0  200.0
2  1.5  30.0  300.0

Transformed DataFrame (impute A, drop D & E, passthrough others):
     A     B      C
0  1.0  10.0  100.0
1  2.0  20.0  200.0
2  1.5  30.0  300.0
"""