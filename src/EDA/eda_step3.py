
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/train_data.csv')

numerical_cols = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
fig, axes = plt.subplots(2,2, figsize=(14,10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col].dropna(), bins=20, color='lightblue', edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].grid(alpha=0.3)
plt.tight_layout()
output_path = 'outputs/numerical_features_dist.png'
plt.savefig(output_path, dpi=300)
plt.close()
plt.show()

for col in numerical_cols:
    print(f'\n--- Statistics for {col} ---')
    print(f' mean of {col}: {df[col].mean()}')
    print (f' median of {col}: {df[col].median()}')
    print(f' standard deviation of {col}: {df[col].std()}')
    print(f' min of {col}: {df[col].min()}')
    print(f' max of {col}: {df[col].max()}')
    print (f' number of unique values in {col}: {df[col].nunique()}')
    print(f' number of missing values in {col}: {df[col].isnull().sum()}')



