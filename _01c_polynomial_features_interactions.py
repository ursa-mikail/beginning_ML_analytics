# To include "feature interactions" in model: Use PolynomialFeatures.
# P.S. This is impractical if you have lots of features, and unnecessary if you're using a tree-based model (it is already doing that).
# use a feature evaluation model to know if you should keep that feature

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Creating a sample dataframe with three features
X = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 4, 4],
    'C': [0, 10, 100]
})

# Display the original data
print("Original DataFrame:")
print(X)

# Apply PolynomialFeatures to include interaction features, but without the bias term (the constant 1).
# interaction_only=True ensures we only generate interaction terms (i.e., no squared terms like A^2 or B^2).
poly = PolynomialFeatures(include_bias=False, interaction_only=True)

# Transform the data to include interaction terms
X_poly = poly.fit_transform(X)

# Convert the resulting numpy array back into a DataFrame for better readability
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

# Display the transformed DataFrame with interaction terms
print("\nTransformed DataFrame with Interaction Features:")
print(X_poly_df)

"""
Original DataFrame:
   A  B    C
0  1  4    0
1  2  4   10
2  3  4  100

Transformed DataFrame with Interaction Features:
     A    B      C   A B    A C    B C
0  1.0  4.0    0.0   4.0    0.0    0.0
1  2.0  4.0   10.0   8.0   20.0   40.0
2  3.0  4.0  100.0  12.0  300.0  400.0

array([[  1.,   4.,   0.,   4.,   0.,   0.],
       [  2.,   4.,  10.,   8.,  20.,  40.],
       [  3.,   4., 100.,  12., 300., 400.]])

A, B, C: These are the original features.
A*B: Interaction term between feature A and feature B.
A*C: Interaction term between feature A and feature C.
B*C: Interaction term between feature B and feature C.       
"""       
