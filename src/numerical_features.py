# src/numerical_features.py
"""
Day 3 - Task 3.1: Numerical Feature Engineering
Purpose: Process numerical columns and create interaction features
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class NumericalFeatureEngineer:
    """Engineer numerical features from raw data"""

    def __init__(self):
        self.numerical_cols = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
        self.feature_cols = []

    def normalize_column(self, series):
        """Scale values to 0-1 range using min-max normalization"""
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return series * 0  # All same value
        return (series - min_val) / (max_val - min_val)

    def engineer_features(self, df):
        """Create all numerical features"""
        print("Engineering numerical features...")

        # Store original columns for reference
        original_cols = df[self.numerical_cols].copy()

        # 1. Normalized versions (0-1 scale)
        for col in self.numerical_cols:
            norm_col = f'{col}_normalized'
            df[norm_col] = self.normalize_column(df[col].fillna(df[col].median()))
            self.feature_cols.append(norm_col)
            print(f"   Created: {norm_col}")

        # 2. Interaction features
        # Sleep-Energy Ratio
        df['sleep_energy_ratio'] = df['sleep_hours'].fillna(df['sleep_hours'].median()) / \
                                   (df['energy_level'].fillna(df['energy_level'].median()) + 0.001)
        self.feature_cols.append('sleep_energy_ratio')
        print(f"   Created: sleep_energy_ratio")

        # Stress-Energy Difference
        df['stress_energy_diff'] = df['stress_level'].fillna(df['stress_level'].median()) - \
                                   df['energy_level'].fillna(df['energy_level'].median())
        self.feature_cols.append('stress_energy_diff')
        print(f"   Created: stress_energy_diff")

        # Energy-Stress Balance
        df['energy_stress_balance'] = df['energy_level'].fillna(df['energy_level'].median()) - \
                                      df['stress_level'].fillna(df['stress_level'].median())
        self.feature_cols.append('energy_stress_balance')
        print(f"   Created: energy_stress_balance")

        # 3. Missing value indicators
        df['sleep_hours_missing'] = df['sleep_hours'].isna().astype(int)
        self.feature_cols.append('sleep_hours_missing')
        print(f"   Created: sleep_hours_missing")

        # 4. Duration per hour (session intensity)
        df['duration_per_hour'] = df['duration_min'].fillna(df['duration_min'].median()) / 60.0
        self.feature_cols.append('duration_per_hour')
        print(f"   Created: duration_per_hour")

        print(f"\nCreated {len(self.feature_cols)} numerical features")

        return df

    def get_feature_columns(self):
        """Return list of engineered numerical feature columns"""
        return self.feature_cols


# Test the feature engineer
if __name__ == "__main__":
    df = pd.read_csv('data/train_data.csv')

    print(f"Original DataFrame shape: {df.shape}")
    print(f"Original numerical columns: {['duration_min', 'sleep_hours', 'energy_level', 'stress_level']}")

    # Engineer features
    engineer = NumericalFeatureEngineer()
    df = engineer.engineer_features(df)

    print(f"\nNew DataFrame shape: {df.shape}")
    print(f"Engineered numerical features: {engineer.get_feature_columns()}")

    print("\nSample statistics for new features:")
    for col in engineer.get_feature_columns()[:5]:  # Show first 5
        print(f"   {col}:")
        print(f"      Mean: {df[col].mean():.3f}")
        print(f"      Std: {df[col].std():.3f}")
        print(f"      Min: {df[col].min():.3f}, Max: {df[col].max():.3f}")
