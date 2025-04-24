import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Define the pipeline without OutlierRemover
numeric_features = list(range(X.shape[1]))  # Use integer indices
# imputer fill any remaining missing values with the mean strategy
numeric_transformer = Pipeline(steps=[
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

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, C=1.0, penalty='l2'))
])

# Define the hyperparameter grid for grid search
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

# Create GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Fit the model with grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Make predictions on the test data using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the performance of the best model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f'Best Hyperparameters: {best_params}')
print(f'Accuracy with Best Model: {accuracy:.2f}')
print('Classification Report:\n', classification_report_str)
print('Confusion Matrix:\n', conf_matrix)
