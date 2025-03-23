import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display basic information
print("Dataset Information:")
print(df.info())
print("\nFirst five rows:")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualizations
sns.set(style="whitegrid")

# Pairplot to visualize relationships
sns.pairplot(df, hue='species', markers=['o', 's', 'D'])
plt.show()

# Boxplot for feature distribution
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='species', y='sepal length (cm)')
plt.title("Sepal Length Distribution by Species")
plt.show()

# Histogram of all features
df.hist(figsize=(10, 8), bins=20)
plt.suptitle("Feature Distributions")
plt.show()

# Observations
print("\nObservations:")
print("1. Setosa species has distinctly smaller sepal and petal sizes.")
print("2. Versicolor and Virginica have overlapping characteristics but differ in petal width and length.")
print("3. No missing values detected in the dataset.")
