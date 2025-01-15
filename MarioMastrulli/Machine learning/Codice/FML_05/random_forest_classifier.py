import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the data
diabetes = pd.read_csv('../data/diabetes.csv')

# Divide features and target variable transforming them into matrices
X = diabetes.drop(['Outcome'], axis=1).values
y = diabetes['Outcome'].values

# Split the dataset into training and test sets through hold-out strategy
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)


'''
Do we need data normalization if we use a RANDOM FOREST?
The answer is no. The same reason explained for decision trees holds.
'''


# Create a decision tree classifier
'''
criterion: the function to measure the quality of a split
n_estimators: the number of trees in the forest
min_samples_leaf: sets the minimum number of samples required to be at a leaf node
max_depth:  limits the maximum depth of the decision tree

>> n_estimators ---> more complexity (overfitting)
<< min_samples_leaf & >> max_depth  --->  more complexity (overfitting)
>> min_samples_leaf & << max_depth  --->  less complexity (underfitting)
'''

# Change hyperparameters as you wish
criterion = 'entropy'
min_samples_leaf = 2
max_depth = 5
clf = RandomForestClassifier(n_estimators=200, criterion=criterion, min_samples_leaf=min_samples_leaf,
                             max_depth=max_depth, random_state=42)

# Train the classifier
clf = clf.fit(X_train, y_train)

# Compute predictions
y_pred = clf.predict(X_test)
print(f"Accuracy TEST: {accuracy_score(y_test, y_pred)}")
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Compute predictions on the training set and evaluating the model on such predictions
# Just for observing overfitting/underfitting
y_pred_train = clf.predict(X_train)
print(f"Accuracy TRAINING: {accuracy_score(y_train, y_pred_train)}")



