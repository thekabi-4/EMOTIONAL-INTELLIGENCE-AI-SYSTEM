# src/data_validation.py
"""
Data Validation Module
Purpose: Validate and report on the training data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path('outputs').mkdir(exist_ok=True)

with open("outputs/data_validation.md", "w") as f:
    f.write("# Data Validation Report\n\n")

    data_path = 'data/train_data.csv'
    df = pd.read_csv(data_path)

    print(df.head())

    f.write("## Dataset Overview\n\n")

    print(f"Dataset loaded: {len(df)} rows and {len(df.columns)} columns")
    f.write(f"- **Total Rows**: {len(df)}\n- **Total Columns**: {len(df.columns)}\n")

    print("\nMissing values:", df.isnull().sum())
    f.write(f"- **Missing Values**: \n```\n{df.isnull().sum()}\n```\n")

    print(f"\nEmotional states: {df['emotional_state'].nunique()}")
    print(f"Intensity range: [{df['intensity'].min()} to {df['intensity'].max()}]")
    print(f"\nData types:\n{df.dtypes}")

    f.write(f"- **Emotional States**: {df['emotional_state'].nunique()}\n")
    f.write(f"- **Intensity Range**: [{df['intensity'].min()} to {df['intensity'].max()}]\n")
    f.write(f"- **Data Types**:\n```\n{df.dtypes}\n```\n")

    print("\nUnique previous_day_mood:", df['previous_day_mood'].unique())
    print("\nUnique time_of_day:", df['time_of_day'].unique())
    print("\nUnique ambience_type:", df['ambience_type'].unique())
    print("\nPrevious_day_mood:", df['previous_day_mood'].unique())
    print("\nFace_emotion_hint:", df['face_emotion_hint'].unique())
    print("\nEmotional_state:", df['emotional_state'].unique())
    print("\nReflection_quality:", df['reflection_quality'].unique())

    f.write(f"- **Unique Previous Day Mood**: {df['previous_day_mood'].unique()}\n")
    f.write(f"- **Unique Time of Day**: {df['time_of_day'].unique()}\n")
    f.write(f"- **Unique Ambience Type**: {df['ambience_type'].unique()}\n")
    f.write(f"- **Unique Face Emotion Hint**: {df['face_emotion_hint'].unique()}\n")
    f.write(f"- **Unique Emotional State**: {df['emotional_state'].unique()}\n")
    f.write(f"- **Unique Reflection Quality**: {df['reflection_quality'].unique()}\n")
