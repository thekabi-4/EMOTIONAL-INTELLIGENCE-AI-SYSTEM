# src/feature_fusion.py
"""
Day 3 - Task 3.3: Feature Fusion Engine
Purpose: Combine text + numerical + categorical features into one matrix
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class FeatureFusion:
    """Combine all feature sources into one unified feature matrix"""

    def __init__(self):
        # Text features (from Day 2)
        self.text_features = [
            'char_count', 'word_count', 'is_short',
            'compound', 'pos', 'neu', 'neg',
            'ambiguity_score', 'coherence_score', 'complexity_score', 'emotion_word_count'
        ]

        # Numerical features (from Day 3 Task 3.1)
        self.numerical_features = [
            'duration_min_normalized', 'sleep_hours_normalized',
            'energy_level_normalized', 'stress_level_normalized',
            'sleep_energy_ratio', 'stress_energy_diff',
            'energy_stress_balance', 'sleep_hours_missing', 'duration_per_hour'
        ]

        # Categorical features (from Day 3 Task 3.2) - will be auto-detected
        self.categorical_features = []

        # Target columns
        self.target_cols = ['emotional_state', 'intensity']

        # Track all feature columns
        self.all_features = []

    def detect_categorical_features(self, df):
        """Auto-detect one-hot encoded categorical columns"""
        print("Detecting categorical features...")

        categorical_features = []

        for col in df.columns:
            # Look for one-hot encoded columns (prefix_pattern)
            prefixes = ['ambience_type_', 'time_of_day_', 'previous_day_mood_',
                       'face_emotion_hint_', 'reflection_quality_']

            if any(col.startswith(prefix) for prefix in prefixes):
                categorical_features.append(col)

        self.categorical_features = categorical_features
        print(f"   Found {len(categorical_features)} categorical features")

        return categorical_features

    def fuse(self, df, include_targets=True):
        """Combine all features into one matrix"""
        print("\nFusing all features...")

        # Auto-detect categorical features
        self.detect_categorical_features(df)

        # Combine all feature lists
        self.all_features = self.text_features + self.numerical_features + self.categorical_features

        print(f"\nFeature Summary:")
        print(f"   Text features: {len(self.text_features)}")
        print(f"   Numerical features: {len(self.numerical_features)}")
        print(f"   Categorical features: {len(self.categorical_features)}")
        print(f"   Total features: {len(self.all_features)}")

        # Check which features exist in DataFrame
        missing_features = [f for f in self.all_features if f not in df.columns]
        if missing_features:
            print(f"\nWarning: Missing features ({len(missing_features)}): {missing_features[:5]}...")

        # Keep only features that exist
        available_features = [f for f in self.all_features if f in df.columns]

        # Create feature matrix (X)
        X = df[available_features].copy()

        # Create target vector (y) if targets exist and requested
        y = None
        if include_targets:
            available_targets = [t for t in self.target_cols if t in df.columns]
            if available_targets:
                y = df[available_targets].copy()
                print(f"\nTargets included: {available_targets}")

        print(f"\nFeature matrix shape: {X.shape}")
        if y is not None:
            print(f"Target matrix shape: {y.shape}")

        return X, y

    def get_feature_columns(self):
        """Return list of all feature column names"""
        return self.all_features

    def get_feature_summary(self):
        """Return detailed summary of features"""
        return {
            'text_features': self.text_features,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'total_features': len(self.all_features),
            'all_features': self.all_features
        }


# Test the fusion engine
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/train_data.csv')
    print(f"Original DataFrame shape: {df.shape}\n")

    # Step 1: Run text pipeline
    print("=" * 60)
    print("STEP 1: Text Features")
    print("=" * 60)
    from text_pipeline import TextPipeline
    text_pipeline = TextPipeline()
    df = text_pipeline.process(df)

    # Step 2: Run numerical feature engineering
    print("\n" + "=" * 60)
    print("STEP 2: Numerical Features")
    print("=" * 60)
    from numerical_features import NumericalFeatureEngineer
    num_engineer = NumericalFeatureEngineer()
    df = num_engineer.engineer_features(df)

    # Step 3: Run categorical encoding
    print("\n" + "=" * 60)
    print("STEP 3: Categorical Features")
    print("=" * 60)
    from categorical_features import CategoricalFeatureEncoder
    cat_encoder = CategoricalFeatureEncoder()
    df = cat_encoder.encode_columns(df, auto_detect=True)

    # Step 4: Fuse all features
    print("\n" + "=" * 60)
    print("STEP 4: Feature Fusion")
    print("=" * 60)
    fusion = FeatureFusion()
    X, y = fusion.fuse(df, include_targets=True)

    # Display results
    print(f"\nFeature Summary:")
    summary = fusion.get_feature_summary()
    print(f"   Total features: {summary['total_features']}")
    print(f"   Text: {len(summary['text_features'])}")
    print(f"   Numerical: {len(summary['numerical_features'])}")
    print(f"   Categorical: {len(summary['categorical_features'])}")

    print(f"\nSample feature matrix (first 5 rows, first 10 columns):")
    print(X.iloc[:5, :10])

    print(f"\nSample targets (first 5 rows):")
    print(y.head() if y is not None else "No targets")

    print(f"\nReady for model training!")
    print(f"   X shape: {X.shape} (samples x features)")
    print(f"   y shape: {y.shape} (samples x targets)" if y is not None else "")
