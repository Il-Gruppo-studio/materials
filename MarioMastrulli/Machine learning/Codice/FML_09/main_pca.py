import numpy as np  # linear algebra
import pandas as pd  # read and wrangle dataframes
import matplotlib.pyplot as plt  # visualization
import seaborn as sns  # statistical visualizations and aesthetics
from sklearn.preprocessing import (FunctionTransformer, StandardScaler)  # preprocessing
from sklearn.decomposition import PCA  # dimensionality reduction
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV,
                                     learning_curve)  # model selection modules
import warnings
from sklearn.svm import SVC
from utils import plot_skew, plot_learning_curve, outlier_hunt

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

df = pd.read_csv('./data/glass.csv')
features = df.columns[:-1].tolist()
print(df.shape)

print(df.head(5))

print(df['Type'].value_counts())

# We want to plot the Skewness value: ideally we want this value close to 0.
# Skewness is a measure of asymmetry or distortion of symmetric distribution.
# It measures the deviation of the given distribution of a random variable from a symmetric distribution,
# such as normal distribution. A normal distribution is without any skewness, as it is symmetrical on
# both sides. Hence, a curve is regarded as skewed if it is shifted towards the right or the left.
plot_skew(df[features])

print(f'Dataset has {len(outlier_hunt(df[features]))} elements as outliers')

plt.figure(figsize=(8,6))
sns.boxplot(df[features])
plt.show()

corr = df[features].corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2',
            annot_kws={'size': 15}, xticklabels=features,
            yticklabels=features, alpha=0.7, cmap='coolwarm')
plt.show()
#
outlier_indices = outlier_hunt(df[features])
df = df.drop(outlier_indices).reset_index(drop=True)
print(df.shape)

plot_skew(df[features])

print(df['Type'].value_counts())

X = df[features]
y = df['Type']

test_size = 0.2
seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# We identify how many components are needed to contain a variance above a threshold of about 80%.
# For this purpose we use a comulative sum to identify the optimal value.
pca = PCA(n_components=X_train.shape[1], random_state=seed)
pca.fit(X_train)
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize=(8,6))
plt.bar(range(1,len(cum_var_exp)+1), var_exp, align= 'center', label= 'individual variance explained', \
       alpha = 0.7)
plt.step(range(1,len(cum_var_exp)+1), cum_var_exp, where = 'mid' , label= 'cumulative variance explained', \
        color= 'red')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.xticks(np.arange(1,len(var_exp)+1,1))
plt.legend(loc='center right')
plt.show()

# Cumulative variance explained
for i, sum in enumerate(cum_var_exp):
    print("PC" + str(i+1), f"Cumulative variance: {cum_var_exp[i]*100} %")

# Pay attention to the order! It is important to perform normalization first and then
# feature selection based on the components identified with PCA.

pca = PCA(n_components=5, random_state=seed)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
params = {
    'C': [10, 50, 100],
    'kernel': ('linear', 'rbf', 'sigmoid'),
    'tol': [1e-2, 1e-3, 1e-4]
}
grid = GridSearchCV(estimator=SVC(random_state=seed), param_grid=params, cv=kfold,
             scoring='accuracy', verbose=1, n_jobs=-1)

grid.fit(X_train, y_train)

print('--------BEST SCORE----------')
print(grid.best_score_*100)
print('--------BEST PARAM----------')
print(grid.best_params_)

plt.figure(figsize=(9,6))

train_sizes, train_scores, test_scores = learning_curve(
              estimator= grid.best_estimator_ , X= X_train, y = y_train,
                train_sizes=np.arange(0.1,1.1,0.1), cv=kfold,  scoring='accuracy', n_jobs= - 1)

plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning curve for SVC')