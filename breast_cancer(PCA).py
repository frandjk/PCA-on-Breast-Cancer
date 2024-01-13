import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('breast_cancer.csv')
df.head(10)
df.shape
df.info()

df.drop(['Unnamed: 32', 'id'], axis = 1, inplace = True)
df.head(10)

# convert categorical data to numreic data
df['diagnosis'] = [
    1 if item == 'M'
    else 0 for item in df['diagnosis']
]
df['diagnosis']

plt.pie(df.diagnosis.value_counts(), startangle = 90, explode = [0.05, 0.05], autopct = '%0.2f%%',
        labels = ['Benign', 'Malignant'], colors = ['lightgreen', 'red'], radius = 2)
plt.show()

# correlation matrix
f, ax = plt.subplots(figsize = (14, 12))
sns.heatmap(df.corr(), cmap = 'bwr', annot = True, linewidths = 0.5, fmt = '.1f', ax = ax)
plt.show()

# PCA 
X = df.drop('diagnosis', axis = 1)
Y = df['diagnosis']

sc = StandardScaler()
X = sc.fit_transform(X)
X.shape

n_components = 3
pca = PCA(n_components = n_components)
pca.fit(X)
components = pca.transform(X)
X.shape

components.shape

explained_var_ratio = pca.explained_variance_ratio_
# np.cumsum(pca.explained_variance_ratio_)

df_pca = pd.DataFrame({'PC': ['PC1', 'PC2', 'PC3'],
                       'var': explained_var_ratio})
df_pca

# PCA visualization
sns.barplot(x = 'PC', y = 'var', data = df_pca, color = 'lightgreen')
plt.ylabel('Variance Explained')
plt.xlabel('Principal Components')
plt.show()

fig = plt.figure(figsize = (15, 8))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c = df['diagnosis'], s = 60)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.view_init(30, 120)

plt.figure(figsize = (18, 8))
plt.subplot(1, 3, 1)
sns.scatterplot( x = df.iloc[:, 0], y = df.iloc[:, 2], hue = df['diagnosis'], palette = 'Set1')
plt.xlabel('PC1')
plt.ylabel('PC3')

plt.subplot(1, 3, 2)
sns.scatterplot(x = df.iloc[:, 1], y = df.iloc[:, 2], hue = df['diagnosis'], palette = 'Set1')
plt.xlabel('PC2')
plt.ylabel('PC3')

plt.subplot(1, 3, 3)
sns.scatterplot(x = df.iloc[:, 0], y = df.iloc[:, 1], hue = df['diagnosis'], palette = 'Set1')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

pca = PCA(n_components = 3)
pca.fit(df)

df_pc = pd.DataFrame(pca.components_, columns = df.columns)
df_pc

plt.figure(figsize = (15, 8))
sns.heatmap(df_pc, cmap = 'viridis')
plt.title('Principal Components Correlation with the Features')
plt.xlabel('Features')
plt.ylabel('Principal Components')

