
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd     
import matplotlib.pyplot as plt

df=pd.read_csv('data/train_data.csv')

state_counts=df['emotional_state'].value_counts()
plt.figure(figsize=(10,6))
plt.bar(state_counts.index, state_counts.values)
plt.xlabel('Emotional State')
plt.ylabel('Count') 
plt.title('Distribution of Emotional States')
plt.xticks(rotation=45)

for i,v in enumerate(state_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda_emotional_state_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
plt.show()

print(f"\nDistribution of emotional states")
for state, count in state_counts.items():
    pct= round (count/len(df)*100,1)
    print(f"{state}: {count} samples ({pct}%)")


