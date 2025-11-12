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
print(df.head())

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

# ============================================================================
# 1. Data Overview
# ============================================================================

print(f"Shape: {df.shape}")
print(f"Time period: {df['year'].min()}-{df['year'].max()}")
print(f"Categories: {df['cat'].nunique()}")

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nSummary statistics:")
narrative_cols = ['Final AGENT', 'Final EVENTS', 'Final WORLD', 'Final Composite']
print(df[narrative_cols].describe())

# ============================================================================
# 2. Distributions
# ============================================================================

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

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. Correlations
# ============================================================================

correlation_matrix = df[narrative_cols].corr()
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=2)
plt.title('Correlation Between Narrative Dimensions')
plt.tight_layout()
plt.savefig('eda_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. Year Trends
# ============================================================================

yearly_means = df.groupby('year')[narrative_cols].mean()

plt.figure(figsize=(12, 6))
for col, color, label in zip(narrative_cols[:3], colors[:3], 
                              ['AGENTS', 'EVENTS', 'WORLD']):
    plt.plot(yearly_means.index, yearly_means[col], 
             marker='o', linewidth=2, label=label, alpha=0.8)

plt.xlabel('Year')
plt.ylabel('Average Score')
plt.title('Narrativity Over Time')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('eda_temporal.png', dpi=300, bbox_inches='tight')
plt.show()

# Test for trend
slope, _, r_val, p_val, _ = linregress(yearly_means.index, 
                                        yearly_means['Final Composite'])
print(f"\nLinear trend test (Composite score):")
print(f"  Slope: {slope:.4f} per year")
print(f"  R²: {r_val**2:.3f}")
print(f"  P-value: {p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'})")

# ============================================================================
# 5. Genre Comparison
# ============================================================================

genre_means = df.groupby('cat')['Final Composite'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
genre_means.plot(kind='barh', color='steelblue', edgecolor='black')
plt.xlabel('Average Composite Score')
plt.title('Narrativity by Genre')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_genres.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. Inter-Annotator Agreement
# ============================================================================

std_cols = ['Agents std', 'Events std', 'World std']
mean_cols = ['Agents mean', 'Events mean', 'World mean']

for col in std_cols:
    print(f"  {col}: {df[col].mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot of disagreement
df[std_cols].boxplot(ax=axes[0], labels=['AGENTS', 'EVENTS', 'WORLD'])
axes[0].set_title('Annotator Disagreement by Dimension')
axes[0].set_ylabel('Standard Deviation')
axes[0].grid(alpha=0.3)

# Mean vs Std scatter
for mean_col, std_col, color, label in zip(mean_cols, std_cols, 
                                            ['blue', 'orange', 'green'],
                                            ['AGENTS', 'EVENTS', 'WORLD']):
    axes[1].scatter(df[mean_col], df[std_col], alpha=0.5, label=label, s=30)

axes[1].set_xlabel('Mean Score')
axes[1].set_ylabel('Standard Deviation')
axes[1].set_title('Score vs Disagreement')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_agreement.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. Summary
# ============================================================================


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


# ============================================================================
# Dimensionality Reduction with PCA
# ============================================================================

# ============================================================================
# 1. Scaling and PCA Model Fitting
# ============================================================================

#  Narrative dimensions
pca_features = ['Final AGENT', 'Final EVENTS', 'Final WORLD']
X_final = df[pca_features].dropna()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# ============================================================================
# 2. Variance Explained by Principal Components
# ============================================================================

# Explained variance
explained_variance = pca.explained_variance_ratio_
#print(explained_variance)
cumulative_variance = np.cumsum(explained_variance)
#print(cumulative_variance)

# ============================================================================
# 2. Scree Plots
# ============================================================================

# ============================================================================
# 3. Bi Plots
# ============================================================================