# src/categorical_features.py
"""
Day 3 - Task 3.2: Categorical Feature Encoding
Purpose: Automatically detect and encode all categorical columns
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class CategoricalFeatureEncoder:
    """Encode categorical columns using one-hot encoding"""

    def __init__(self, exclude_cols=None, max_unique=10):
        """
        Initialize encoder

        Args:
            exclude_cols: List of columns to exclude from encoding
            max_unique: Maximum unique values to consider as categorical
        """
        self.exclude_cols = exclude_cols or [
            'journal_text', 'cleaned_text', 'emotional_state',
            'intensity', 'sentiment'  # Text and target columns
        ]
        self.max_unique = max_unique
        self.categorical_cols = []
        self.encoded_cols = []

    def detect_categorical_columns(self, df):
        """Automatically detect categorical columns in DataFrame"""
        print("Detecting categorical columns...")

        categorical_cols = []

        for col in df.columns:
            # Skip excluded columns
            if col in self.exclude_cols:
                continue

            # Check if column is object/string type (both old and new pandas)
            if df[col].dtype == 'object' or str(df[col].dtype) == 'string' or str(df[col].dtype) == 'str':
                # Check if it has low cardinality
                unique_count = df[col].nunique()
                if unique_count <= self.max_unique:
                    categorical_cols.append(col)
                    print(f"   Detected: {col} ({unique_count} unique values)")

        self.categorical_cols = categorical_cols
        print(f"\nFound {len(categorical_cols)} categorical columns to encode")

        return categorical_cols

    def encode_columns(self, df, auto_detect=True):
        """Apply one-hot encoding to categorical columns"""
        print("\nEncoding categorical features...")

        # Auto-detect or use predefined columns
        if auto_detect:
            self.detect_categorical_columns(df)

        if len(self.categorical_cols) == 0:
            print("   Warning: No categorical columns found to encode")
            return df

        for col in self.categorical_cols:
            if col not in df.columns:
                print(f"   Warning: Column {col} not found, skipping")
                continue

            # Handle missing values by creating 'unknown' category
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna('unknown')

            if missing_count > 0:
                print(f"   {col}: {missing_count} missing values -> 'unknown' category")

            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)

            # Add dummy columns to DataFrame
            df = pd.concat([df, dummies], axis=1)

            # Drop original column
            df = df.drop(columns=[col])

            # Track new column names
            new_cols = list(dummies.columns)
            self.encoded_cols.extend(new_cols)

            print(f"   Encoded {col}: {len(new_cols)} categories")

        print(f"\nCreated {len(self.encoded_cols)} categorical features")

        return df

    def get_feature_columns(self):
        """Return list of encoded categorical feature columns"""
        return self.encoded_cols

    def get_encoding_summary(self):
        """Return summary of encoding process"""
        return {
            'categorical_cols_detected': self.categorical_cols,
            'total_encoded_features': len(self.encoded_cols),
            'encoded_columns': self.encoded_cols
        }


# Test the encoder
if __name__ == "__main__":
    df = pd.read_csv('data/train_data.csv')

    print(f"Original DataFrame shape: {df.shape}")
    print(f"\nAll columns and their types:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype} ({df[col].nunique()} unique)")

    # Encode features with auto-detection
    encoder = CategoricalFeatureEncoder()
    df = encoder.encode_columns(df, auto_detect=True)

    print(f"\nNew DataFrame shape: {df.shape}")
    print(f"\nEncoded categorical features ({len(encoder.get_feature_columns())} total):")

    # Show all encoded columns
    for col in encoder.get_feature_columns():
        print(f"   - {col}")

    print("\nEncoding Summary:")
    summary = encoder.get_encoding_summary()
    print(f"   Categorical columns detected: {summary['categorical_cols_detected']}")
    print(f"   Total encoded features: {summary['total_encoded_features']}")

    print("\nVerify one-hot encoding:")
    sample_row = df.iloc[0]
    ambience_cols = [c for c in df.columns if c.startswith('ambience_type_')]
    active = [c for c in ambience_cols if sample_row[c] == 1]
    print(f"   Row 0 ambience: {active[0] if active else 'none'} = 1")
