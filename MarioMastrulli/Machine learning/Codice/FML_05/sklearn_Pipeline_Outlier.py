import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bound = []
        self.upper_bound = []

    def outlier_detector(self, X):
        # Calculate quartiles
        q1 = np.percentile(X, 25)
        q3 = np.percentile(X, 75)

        # Calculate IQR (Interquartile Range)
        iqr = q3 - q1

        # Calculate lower and upper bounds to identify outliers
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        # Initialize lower and upper bounds
        self.lower_bound = []
        self.upper_bound = []

        # Apply the outlier_detector function along axis 0 (columns)
        np.apply_along_axis(self.outlier_detector, axis=0, arr=X)

        return self

    def transform(self, X, y=None):
        # Copy the input array to avoid unwanted changes
        X = np.copy(X)

        # Iterate over all columns
        for i in range(X.shape[1]):
            x = X[:, i]

            # Masks to identify outliers
            lower_mask = x < self.lower_bound[i]
            upper_mask = x > self.upper_bound[i]

            # Set values that are considered outliers to NaN
            x[lower_mask | upper_mask] = np.nan

            # Assign the transformed column back to the original array
            X[:, i] = x

        # Impute NaN values with the mean
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        return X

# Load the diabetes dataset
diabetes = pd.read_csv('diabetes.csv')

# Print dataset stats
print(diabetes.describe())
print(diabetes.columns)

# Shuffling all samples to avoid group bias
diabetes = diabetes.sample(frac=1).reset_index(drop=True)

# Select features and target variable
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes[selected_features].values
y = diabetes['Outcome'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline with the custom OutlierRemover (list of index from 0 to X.shape[1] - 1)
numeric_features = list(range(X.shape[1]))  # Use integer indices

# imputer fill any remaining missing values with the mean strategy
numeric_transformer = Pipeline(steps=[
    ('outlier_remover', OutlierRemover()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
"""
1) Numeric_transformer processed features replace 
the original numerical features in the pipeline, retaining only the modifications.
2) remainder = passthrough, the features not involved in the transformations 
are included in the output without undergoing any modification."""

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='passthrough'
)
# Preprocessor manages the removal of outliers, imputation, and standardization of numerical features
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, C=1.0, penalty='l2'))
])

# Fit the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report_str)
print('Confusion Matrix:\n', conf_matrix)
