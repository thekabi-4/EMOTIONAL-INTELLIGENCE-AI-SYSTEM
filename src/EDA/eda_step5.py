
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train_data.csv')

df["char_cont"]= df["journal_text"].apply(lambda x: len(str(x)))
df["word_cont"]= df["journal_text"].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1,2, figsize=(14,6))

axes[0].hist(df["char_cont"], bins=30, color='lightgreen', edgecolor='black')
axes[0].set_title('Distribution of Character Count')
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].grid(alpha=0.3)

axes[1].hist(df["word_cont"], bins=30, color='lightcoral', edgecolor='black')
axes[1].set_title('Distribution of Word Count')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/text_length_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n--- Text Length Analysis ---")
print(f"Characters mean : {df['char_cont'].mean():.2f}")
print(f"Words mean : {df['word_cont'].mean():.2f}")
print(f"Characters median : {df['char_cont'].median():.2f}")
print(f"Words median : {df['word_cont'].median():.2f}")
print(f"Characters std : {df['char_cont'].std():.2f}")
print(f"Words std : {df['word_cont'].std():.2f}")
print(f"Characters min : {df['char_cont'].min()}")
print(f"Words min : {df['word_cont'].min()}")   
print(f"Characters max : {df['char_cont'].max()}")
print(f"Words max : {df['word_cont'].max()}")


