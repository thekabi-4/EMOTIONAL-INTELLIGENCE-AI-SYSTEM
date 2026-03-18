import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/train_data.csv')

cols_for_correlation = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level', 'intensity']
corr_matrix = df[cols_for_correlation].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    center=0, 
    fmt='.2f'
)

plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
output_path = 'outputs/correlation_matrix.png'
plt.savefig(output_path, dpi = 300, bbox_inches = 'tight')
plt.close()


print("\n--- Correlation Matrix ---")
for col in cols_for_correlation:
    if col != 'intensity':
        corr_val = corr_matrix.loc[col, 'intensity']
        strength = "strong" if abs(corr_val) > 0.5 else "moderate" if abs(corr_val) > 0.3 else "weak"
        direction = "positive" if corr_val > 0 else "negative"
        print(f"   {col}: {corr_val:.3f} ({strength} {direction})")