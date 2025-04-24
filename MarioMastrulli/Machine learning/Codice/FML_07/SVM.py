from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Load dataset
data = datasets.load_iris(as_frame=True)
X = data.data
y = data.target

# Add some missing values
X = X.copy()
X.iloc[::10, 0] = np.nan

# Drop rows with missing values
X = X.dropna()
y = y.loc[X.index]

# Separate continuous and categorical features
continuous_features = X.select_dtypes(include=["float64", "int64"]).columns
categorical_features = X.select_dtypes(exclude=["float64", "int64"]).columns

# Scale only continuous features
scaler = StandardScaler()
X_continuous_scaled = scaler.fit_transform(X[continuous_features])

# Combine scaled continuous features and categorical features (if any)
X_preprocessed = pd.DataFrame(X_continuous_scaled, columns=continuous_features)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Train an SVM
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))
