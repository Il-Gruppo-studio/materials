 import numpy as np
import pandas as pd
from utilities import plot_theta_gd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('datasets/diabetes.csv')
print(df.describe())

features_names, label_name = df.columns[:-1], df.columns[-1]

df = df.sample(frac=1).reset_index(drop=True)

x = df[features_names].values
y = df[label_name].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

logistic_regression = LogisticRegression(random_state=42)
logistic_regression.fit(x_train, y_train)

y_pred = logistic_regression.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
