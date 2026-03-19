
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# src/model_evaluation.py
"""
Day 4 - Task 4.3: Model Evaluation
Purpose: Evaluate both trained models and generate summary report
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import pickle
from pathlib import Path

print("=" * 60)
print("DAY 4 - TASK 4.3: MODEL EVALUATION")
print("=" * 60)

# Step 1: Load preprocessed data (FAST - no processing)
print("\nStep 1: Loading preprocessed data...")
X = pd.read_pickle('data/processed/X_train.pkl')
y = pd.read_pickle('data/processed/y_train.pkl')
print(f"   Feature matrix shape: {X.shape}")
print(f"   Target matrix shape: {y.shape}")

# Step 2: Load trained models
print("\nStep 2: Loading trained models...")

with open('models/emotion_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('models/intensity_regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

print("   Loaded: emotion_classifier.pkl")
print("   Loaded: intensity_regressor.pkl")

# Step 3: Split data (same split as training for fair evaluation)
print("\nStep 3: Preparing test set...")

from sklearn.model_selection import train_test_split

X_train, X_test, y_full_train, y_full_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['emotional_state']
)

y_emotion_test = y_full_test['emotional_state']
y_intensity_test = y_full_test['intensity']

print(f"   Test set: {len(X_test)} samples")

# Step 4: Evaluate classifier
print("\nStep 4: Evaluating Emotional State Classifier...")

y_pred_emotion = classifier.predict(X_test)

classifier_accuracy = accuracy_score(y_emotion_test, y_pred_emotion)
classifier_baseline = 1 / y['emotional_state'].nunique()

print(f"\nClassifier Results:")
print(f"   Accuracy: {classifier_accuracy:.3f} ({classifier_accuracy*100:.1f}%)")
print(f"   Baseline: {classifier_baseline:.3f} ({classifier_baseline*100:.1f}%)")
print(f"   Improvement: +{(classifier_accuracy - classifier_baseline)*100:.1f} points")

# Step 5: Evaluate regressor
print("\nStep 5: Evaluating Intensity Regressor...")

y_pred_intensity_raw = regressor.predict(X_test)
y_pred_intensity = np.round(y_pred_intensity_raw).clip(1, 5).astype(int)

mae = mean_absolute_error(y_intensity_test, y_pred_intensity_raw)
rmse = np.sqrt(np.mean((y_intensity_test - y_pred_intensity_raw) ** 2))
r2 = r2_score(y_intensity_test, y_pred_intensity_raw)
exact_match = (y_pred_intensity == y_intensity_test).mean()

print(f"\nRegressor Results:")
print(f"   MAE: {mae:.3f} (target: <2.0)")
print(f"   RMSE: {rmse:.3f}")
print(f"   R-squared: {r2:.3f}")
print(f"   Exact Match: {exact_match:.3f} ({exact_match*100:.1f}%)")

# Step 6: Combined evaluation
print("\nStep 6: Combined Model Performance...")

both_correct = ((y_pred_emotion == y_emotion_test) & (y_pred_intensity == y_intensity_test)).mean()

print(f"\nCombined Results:")
print(f"   Both predictions correct: {both_correct:.3f} ({both_correct*100:.1f}%)")
print(f"   Classifier only correct: {(y_pred_emotion == y_emotion_test).mean() - both_correct:.3f}")
print(f"   Regressor only correct: {(y_pred_intensity == y_intensity_test).mean() - both_correct:.3f}")
print(f"   Both incorrect: {1 - ((y_pred_emotion == y_emotion_test) | (y_pred_intensity == y_intensity_test)).mean():.3f}")

# Step 7: Generate summary report
print("\nStep 7: Generating evaluation report...")

report = {
    'classifier': {
        'accuracy': classifier_accuracy,
        'baseline': classifier_baseline,
        'improvement': classifier_accuracy - classifier_baseline
    },
    'regressor': {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'exact_match': exact_match
    },
    'combined': {
        'both_correct': both_correct
    },
    'test_size': len(X_test)
}

# Save report
Path('outputs').mkdir(exist_ok=True)
report_df = pd.DataFrame([report])
report_df.to_csv('outputs/model_evaluation_report.csv', index=False)

print("   Report saved: outputs/model_evaluation_report.csv")

# Step 8: Print summary
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

print(f"\nEmotional State Classifier:")
print(f"   Accuracy: {classifier_accuracy*100:.1f}% (baseline: {classifier_baseline*100:.1f}%)")
if classifier_accuracy > classifier_baseline:
    print("   Status: PASS - Beats random baseline")
else:
    print("   Status: FAIL - Does not beat baseline")

print(f"\nIntensity Regressor:")
print(f"   MAE: {mae:.3f} (target: <2.0)")
if mae < 2.0:
    print("   Status: PASS - Meets MAE target")
else:
    print("   Status: FAIL - Does not meet MAE target")

print(f"\nCombined Performance:")
print(f"   Both correct: {both_correct*100:.1f}%")

print("\n" + "=" * 60)
print("TASK 4.3 COMPLETE: Model Evaluation Finished")
print("=" * 60)


