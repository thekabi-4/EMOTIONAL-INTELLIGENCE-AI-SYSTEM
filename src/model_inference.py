
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# src/model_inference.py
"""
Day 4 - Task 4.4: Model Inference Script
Purpose: Load trained models and make predictions on new journal entries
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("=" * 60)
print("DAY 4 - TASK 4.4: MODEL INFERENCE")
print("=" * 60)

# Step 1: Load trained models
print("\nStep 1: Loading trained models...")

with open('models/emotion_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('models/intensity_regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

with open('models/emotion_classifier_features.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

print("   Loaded: emotion_classifier.pkl")
print("   Loaded: intensity_regressor.pkl")
print("   Loaded: feature columns (47 features)")

# Step 2: Load preprocessed data
print("\nStep 2: Loading preprocessed data...")

X = pd.read_pickle('data/processed/X_train.pkl')
y = pd.read_pickle('data/processed/y_train.pkl')

print(f"   Feature matrix shape: {X.shape}")
print(f"   Target matrix shape: {y.shape}")

# Step 3: Make predictions on sample data
print("\nStep 3: Making predictions on test samples...")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['emotional_state']
)

# Take first 10 samples for demonstration
sample_indices = X_test.index[:10]
X_samples = X.loc[sample_indices]
y_samples = y.loc[sample_indices]

# Make predictions
emotion_pred = classifier.predict(X_samples)
emotion_proba = classifier.predict_proba(X_samples)
intensity_pred_raw = regressor.predict(X_samples)
intensity_pred = np.round(intensity_pred_raw).clip(1, 5).astype(int)

# Display results
print("\n" + "-" * 60)
print("PREDICTION RESULTS (10 Sample Journal Entries)")
print("-" * 60)

for i, idx in enumerate(sample_indices):
    actual_emotion = y_samples.loc[idx, 'emotional_state']
    actual_intensity = y_samples.loc[idx, 'intensity']
    
    pred_emotion = emotion_pred[i]
    pred_intensity = intensity_pred[i]
    pred_confidence = emotion_proba[i].max()
    
    emotion_status = "CORRECT" if actual_emotion == pred_emotion else "INCORRECT"
    intensity_status = "CORRECT" if actual_intensity == pred_intensity else "INCORRECT"
    
    print(f"\nSample {i+1}:")
    print(f"   Emotion:  Actual={actual_emotion:12} | Predicted={pred_emotion:12} | {emotion_status}")
    print(f"   Intensity: Actual={actual_intensity}          | Predicted={pred_intensity}          | {intensity_status}")
    print(f"   Confidence: {pred_confidence:.2f}")

# Step 4: Calculate inference metrics
print("\n" + "=" * 60)
print("INFERENCE METRICS")
print("=" * 60)

emotion_accuracy = (emotion_pred == y_samples['emotional_state'].values).mean()
intensity_mae = np.mean(np.abs(intensity_pred_raw - y_samples['intensity'].values))
both_correct = ((emotion_pred == y_samples['emotional_state'].values) & 
                (intensity_pred == y_samples['intensity'].values)).mean()

print(f"\nEmotion Classification Accuracy: {emotion_accuracy*100:.1f}%")
print(f"Intensity MAE: {intensity_mae:.3f}")
print(f"Both Predictions Correct: {both_correct*100:.1f}%")

# Step 5: Save inference results
print("\nStep 4: Saving inference results...")

Path('outputs').mkdir(exist_ok=True)

inference_results = pd.DataFrame({
    'sample_id': range(1, len(sample_indices) + 1),
    'actual_emotion': y_samples['emotional_state'].values,
    'predicted_emotion': emotion_pred,
    'actual_intensity': y_samples['intensity'].values,
    'predicted_intensity': intensity_pred,
    'confidence': emotion_proba.max(axis=1)
})

inference_results.to_csv('outputs/inference_results.csv', index=False)
print("   Saved: outputs/inference_results.csv")

# Step 6: Summary
print("\n" + "=" * 60)
print("TASK 4.4 COMPLETE: Model Inference Ready")
print("=" * 60)

print("\nModels are ready for deployment!")
print("To make predictions on new data:")
print("  1. Preprocess new journal entries using the same pipeline")
print("  2. Load models using pickle.load()")
print("  3. Call classifier.predict() and regressor.predict()")
print("  4. Interpret results with confidence scores")


