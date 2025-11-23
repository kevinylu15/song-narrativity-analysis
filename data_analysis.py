import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress



# Load song narrativity annotations dataset
df = pd.read_csv("C:/Users/lub11/OneDrive/Documents/SI 671/annotations.tsv", sep='\t')
print(df.head()) # first 5 rows

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

# Descriptive Statistics

print(f"Shape: {df.shape}") # rows, columns
print(f"Time period: {df['year'].min()}-{df['year'].max()}") # timeline of data
print("\nMissing values:") 
print(df.isnull().sum()[df.isnull().sum() > 0]) # missing values per column
print("\nSummary statistics:")
narrative_cols = ['Final AGENT', 'Final EVENTS', 'Final WORLD', 'Final Composite']
print(df[narrative_cols].describe()) # descriptive stats

# Histogram plots for each narrative dimension

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = ['blue', 'green', 'orange', 'black']
for idx, (col, color) in enumerate(zip(narrative_cols, colors)):
    ax = axes[idx // 2, idx % 2]
    ax.hist(df[col].dropna(), bins=25, color=color, edgecolor='black', alpha=0.7)
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df[col].mean():.2f}')
    ax.set_title(col.replace('Final ', ''))
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

#plt.savefig('eda_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation Matrix bewteen Narrative Dimensions (Heatmap)

correlation_matrix = df[narrative_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=2)
plt.title('Correlation Between Narrative Dimensions')
plt.tight_layout()
#plt.savefig('eda_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# Yearly Trends in Narrativity (Line Plot)

yearly_means = df.groupby('year')[narrative_cols].mean()
for col, color, label in zip(narrative_cols[:3], colors[:3], ['AGENTS', 'EVENTS', 'WORLD']):
    plt.plot(yearly_means.index, yearly_means[col], 
             marker='o', linewidth=2, label=label, alpha=0.8)

plt.xlabel('Year')
plt.ylabel('Average Score')
plt.title('Narrativity Over Time')
plt.legend()
plt.tight_layout()
#plt.savefig('eda_temporal.png', dpi=300, bbox_inches='tight')
plt.show()

# Seeing if there is a Temporal Trend (Linear Regression)
slope, _, r_val, p_val, _ = linregress(yearly_means.index, yearly_means['Final Composite'])
print(f"\nLinear trend test (Composite score):")
print(f"  Slope: {slope:.4f} per year")
print(f"  R²: {r_val**2:.3f}")
print(f"  P-value: {p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'})")

# Genre Differences in Narrativity (Bar Plot)

genre_means = df.groupby('cat')['Final Composite'].mean().sort_values(ascending=False)
genre_means.plot(kind='barh')
plt.xlabel('Average Composite Score')
plt.title('Narrativity by Genre')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
#plt.savefig('eda_genres.png', dpi=300, bbox_inches='tight')
plt.show()

# Annotator Agreement Analysis

std_cols = ['Agents std', 'Events std', 'World std']
mean_cols = ['Agents mean', 'Events mean', 'World mean']
for col in std_cols:
    print(f"  {col}: {df[col].mean():.3f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df[std_cols].boxplot(ax=axes[0], labels=['AGENTS', 'EVENTS', 'WORLD'])
axes[0].set_title('Annotator Disagreement by Dimension')
axes[0].set_ylabel('Standard Deviation')
axes[0].grid(alpha=0.3)

# Mean vs Std scatter Annotator Disagreement
for mean_col, std_col, color, label in zip(mean_cols, std_cols, ['blue', 'orange', 'green'], ['AGENTS', 'EVENTS', 'WORLD']):
    axes[1].scatter(df[mean_col], df[std_col], alpha=0.5, label=label, s=30)
axes[1].set_xlabel('Mean Score')
axes[1].set_ylabel('Standard Deviation')
axes[1].set_title('Score vs Disagreement')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
# plt.savefig('eda_agreement.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary of EDA Findings
print(f"""
SUMMARY:
1. Dataset: {len(df)} songs from {df['year'].min()}-{df['year'].max()}

2. Distributions: All narrative dimensions approximately normal,
   mean Composite score = {df['Final Composite'].mean():.2f}

3. Correlations: High positive correlations (r > 0.6) between all dimensions
   Suggests presence of narrativity factor

4. Temporal trend: Increasing narrativity over time
   (slope = {slope:.4f}, p = {p_val:.4f})

5. Genre differences: {genre_means.index[0]} highest, {genre_means.index[-1]} lowest

6. Annotator agreement: Normal variability (mean std ≈ {df[std_cols].mean().mean():.2f})

""")

# EDA Conclusion:
# strong inter-dimension correlations, temporal increase, genre effects suggest narrativity is meaningful in music
# Looks like we can collapse the 3 dimensions into a smaller set via dimensionality reduction

# Research question for dimensionality reduction: what underlying factors/principal components explain narrativity
# The goal is to identify key dimensions that capture most variance in narrativity scores across songs.
# With the high correlations, we can explain 



# ============================================================================
# Dimensionality Reduction with PCA
# ============================================================================

# Scaling and Model Fitting

#  Narrative dimensions
pca_features = ['Final AGENT', 'Final EVENTS', 'Final WORLD']
X_final = df[pca_features].dropna()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variance Explained by Principal Components

# Explained variance
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance by Principal Components:")
print(explained_variance)
cumulative_variance = np.cumsum(explained_variance)
print("\nCumulative Explained Variance by Principal Components:")
print(cumulative_variance)

# Scree Plots

# Bi Plots

# Singificant Components Loadings