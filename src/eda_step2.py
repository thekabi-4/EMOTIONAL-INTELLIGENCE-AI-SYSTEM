import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('data/train_data.csv')

intensity_counts = df['intensity'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(intensity_counts.index, intensity_counts.values, color='skyblue', edgecolor='black')
plt.xlabel('Intensity (1=Low, 5=High)')
plt.ylabel('No.of Samples')
plt.title('Distribution of Intensity Levels', fontsize=16, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/eda_intensity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
plt.show()

for i,v in enumerate(intensity_counts.values):
    plt.text(intensity_counts.index[i], v + 5, str(v), ha='center', va='bottom', fontweight='bold')

print(f"\n📊 Intensity Statistics:")
print(f"   Mean: {df['intensity'].mean():.2f}")
print(f"   Median: {df['intensity'].median()}")
print(f"   Std Dev: {df['intensity'].std():.2f}")
print(f"   Min: {df['intensity'].min()}")
print(f"   Max: {df['intensity'].max()}")
print (f"   Unique Levels: {df['intensity'].nunique()}")
print(f"counts:\n{df['intensity'].value_counts()}")