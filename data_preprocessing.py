import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Load the dataset
data = pd.read_csv("C:/Users/lub11/OneDrive/Documents/SI 671/annotations.tsv", sep='\t')
#print(data.head())

# EDA
print(data.describe()) 
#print(data.shape) # 1076 rows
#print(data.isnull().sum()) # missing all 3rd person column values, adjudicated columns around 90% values missing
#print(data.dtypes)

# Select final narrative dimensions
pca_features = ['Final AGENT', 'Final EVENTS', 'Final WORLD']
X_final = data[pca_features].dropna()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
cumulative_variance = np.cumsum(explained_variance)
print(cumulative_variance)