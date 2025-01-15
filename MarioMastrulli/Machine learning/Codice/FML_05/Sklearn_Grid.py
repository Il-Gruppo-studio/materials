import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the diabetes dataset
diabetes = pd.read_csv('diabetes.csv')

# Print dataset stats
print(diabetes.describe())
print(diabetes.columns)

# Shuffling all samples to avoid group bias


# Select features and target variable
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes[selected_features].values
y = diabetes['Outcome'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Create a logistic regression model
logistic_model = LogisticRegression(random_state=42)

# Define the hyperparameter grid for grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2'], # Regularization type
    'tol' : [0.1, 0.001]
}

# Create GridSearchCV object
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy', verbose=True)

# Fit the model with grid search on the standardized training data
grid_search.fit(X_train_std, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Make predictions on the standardized test data using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_std)

# Evaluate the performance of the best model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f'Best Hyperparameters: {best_params}')
print(f'Accuracy with Best Model: {accuracy:.2f}')
print('Classification Report:\n', classification_report_str)
print('Confusion Matrix:\n', conf_matrix)
