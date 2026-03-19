
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# src/preprocess_data.py
"""
Preprocess Data Pipeline
Purpose: Process raw data once and save for future use
"""

import pandas as pd
import pickle
from pathlib import Path

print("=" * 60)
print("DATA PREPROCESSING PIPELINE")
print("=" * 60)

# Step 1: Load raw data
print("\nStep 1: Loading raw data...")
df = pd.read_csv("data/train_data.csv")
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# Step 2: Run feature pipeline
print("\nStep 2: Running feature pipeline...")

from text_pipeline import TextPipeline
text_pipeline = TextPipeline()
df = text_pipeline.process(df)

from numerical_features import NumericalFeatureEngineer
num_engineer = NumericalFeatureEngineer()
df = num_engineer.engineer_features(df)

from categorical_features import CategoricalFeatureEncoder
cat_encoder = CategoricalFeatureEncoder()
df = cat_encoder.encode_columns(df, auto_detect=True)

from feature_fusion import FeatureFusion
fusion = FeatureFusion()
X, y = fusion.fuse(df, include_targets=True)

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Final target vector shape: {y.shape}")

# Step 3: Save preprocessed data
print("\nStep 3: Saving preprocessed data...")
Path('data/processed').mkdir(exist_ok=True)

X.to_pickle('data/processed/X_train.pkl')
y.to_pickle('data/processed/y_train.pkl')
print("Preprocessed data saved to 'data/processed/' directory")

feature_cols = list(X.columns)
with open('data/processed/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("Saved: data/processed/feature_columns.pkl")

with open('data/processed/label_encoders.pkl', 'wb') as f:
    pickle.dump({'emotion_classes': y['emotional_state'].unique()}, f)
print("Saved: data/processed/label_encoders.pkl")

# Step 4: Verify saved files
print("\nStep 4: Verifying saved files...")

X_loaded = pd.read_pickle('data/processed/X_train.pkl')
y_loaded = pd.read_pickle('data/processed/y_train.pkl')

print(f"   X_train.pkl shape: {X_loaded.shape}")
print(f"   y_train.pkl shape: {y_loaded.shape}")

if X_loaded.shape == X.shape and y_loaded.shape == y.shape:
    print("   Verification: SUCCESS")
else:
    print("   Verification: FAILED")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)


