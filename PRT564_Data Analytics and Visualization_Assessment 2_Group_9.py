# importing all the necessary module or libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# loading the given dataset (CSV file) into a dataframe
df = pd.read_csv('retractions35215.csv')

# Data preprocessing steps

# dropping missing values
df.dropna(inplace=True)

# converting non-numeric columns to numeric types or encoding them
# non-numeric columns are dropped for now for simplicity
df_numeric = df.select_dtypes(include=['number'])

# assuming 'CitationCount' is the target variable for multiple linear regression
X = df_numeric.drop('CitationCount', axis=1)
y = df_numeric['CitationCount']

# Multiple Linear Regression

# fitting multiple linear regression model
model = sm.OLS(y, sm.add_constant(X)).fit()

# printing summary of the model
print(model.summary())

# extracting coefficient of determination (r-squared)
r_squared = model.rsquared
print("\nCoefficient of Determination (r-squared):", r_squared)

# extracting p-values for the coefficients
p_values = model.pvalues
print("\nP-values for the coefficients:")
print(p_values)

# regression results is shown via visualisation
plt.figure(figsize=(10, 6))
sns.scatterplot(x=model.predict(), y=y)
plt.xlabel('Predicted Citation Count')
plt.ylabel('Actual Citation Count')
plt.title('Multiple Linear Regression')
plt.show()

# Dimensionality Reduction (Principal Component Analysis)

# applying PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# visualising reduced-dimensional data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()

# K-means Clustering

# applying K-means clustering algorithm
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# visualising clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Visualization')
plt.legend(title='Cluster')
plt.show()

# Feature Scaling

# applying feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ordinary Least Squares Method

# fitting ordinary least squares model
ols_model = sm.OLS(y, sm.add_constant(X_scaled)).fit()

# printing summary of the model
print(ols_model.summary())

# additional visualizations like scatter plot, box-plot are also done

plt.figure(figsize=(10, 6))
sns.countplot(x='ArticleType', data=df)
plt.xlabel('ArticleType')
plt.ylabel('Count')
plt.title('Distribution of ArticleTypes')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='Subject', data=df)
plt.xlabel('Subject')
plt.ylabel('Count')
plt.title('Distribution of Subject')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='Journal', data=df)
plt.xlabel('Journal')
plt.ylabel('Count')
plt.title('Distribution of Journal')
plt.xticks(rotation=45)
plt.show()


# scatter plot for 'CitationCount' vs 'Record ID'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Record ID', y='CitationCount', data=df_numeric)
plt.xlabel('Record ID')
plt.ylabel('Citation Count')
plt.title('Scatter Plot of Citation Count vs Record ID')
plt.show()

# Box plot for 'CitationCount' grouped by 'Article Type'
plt.figure(figsize=(10, 6))
sns.boxplot(x='ArticleType', y='CitationCount', data=df)
plt.xlabel('Article Type')
plt.ylabel('Citation Count')
plt.title('Box Plot of Citation Count by Article Type')
plt.xticks(rotation=45)
plt.show()

# Histogram of 'CitationCount'
plt.figure(figsize=(10, 6))
sns.histplot(data=df_numeric, x='CitationCount', bins=20, kde=True)
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.title('Histogram of Citation Count')
plt.show()
