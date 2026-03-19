
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# src/intensity_regressor.py
"""
Day 4 - Task 4.2: Intensity Regressor
Purpose: Train a model to predict emotional intensity (1-5 scale)
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path

print("=" * 60)
print("DAY 4 - TASK 4.2: INTENSITY REGRESSOR")
print("=" * 60)

# Step 1: Load data and create features
print("\nStep 1: Loading data and creating features...")

df = pd.read_csv('data/train_data.csv')

from feature_fusion import FeatureFusion
from text_pipeline import TextPipeline
from numerical_features import NumericalFeatureEngineer
from categorical_features import CategoricalFeatureEncoder

text_pipeline = TextPipeline()
df = text_pipeline.process(df)

num_engineer = NumericalFeatureEngineer()
df = num_engineer.engineer_features(df)

cat_encoder = CategoricalFeatureEncoder()
df = cat_encoder.encode_columns(df, auto_detect=True)

fusion = FeatureFusion()
X, y = fusion.fuse(df, include_targets=True)

y_intensity = y['intensity']

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y_intensity.shape}")

# Step 2: Split data
print("\nStep 2: Splitting data into train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_intensity, 
    test_size=0.2,
    random_state=42
)

print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

# Step 3: Train regressor
print("\nStep 3: Training Gradient Boosting Regressor...")

regressor = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

print("   Training... (this may take 30-60 seconds)")
regressor.fit(X_train, y_train)
print("   Training complete!")

# Step 4: Evaluate on test set
print("\nStep 4: Evaluating model performance...")

y_pred = regressor.predict(X_test)

# Round predictions to nearest integer (1-5 scale)
y_pred_rounded = np.round(y_pred).clip(1, 5).astype(int)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Accuracy with rounded predictions
accuracy_rounded = (y_pred_rounded == y_test).mean()

print(f"\nMean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R2): {r2:.3f}")
print(f"Exact Match Accuracy (rounded): {accuracy_rounded:.3f} ({accuracy_rounded*100:.1f}%)")

# Baseline comparison
baseline_mae = y_train.std()  # Random baseline MAE
print(f"\nBaseline MAE (std of target): {baseline_mae:.3f}")
print(f"Improvement: {baseline_mae - mae:.3f}")

if mae < 2.0:
    print("   Model meets target (MAE < 2.0)!")
else:
    print("   Model needs improvement (target: MAE < 2.0)")

# Step 5: Feature Importance
print("\nStep 5: Top 10 Most Important Features...")

importances = regressor.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 features:")
for i, row in importance_df.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Step 6: Save the model
print("\nStep 6: Saving trained model...")

Path('models').mkdir(exist_ok=True)

with open('models/intensity_regressor.pkl', 'wb') as f:
    pickle.dump(regressor, f)

with open('models/intensity_regressor_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("   Model saved: models/intensity_regressor.pkl")
print("   Features saved: models/intensity_regressor_features.pkl")

# Step 7: Test on sample predictions
print("\nStep 7: Sample Predictions...")

sample_indices = X_test.index[:5]
for idx in sample_indices:
    actual = y_test.loc[idx]
    predicted_raw = regressor.predict(X.loc[[idx]])[0]
    predicted_rounded = int(round(predicted_raw))
    
    status = "CORRECT" if actual == predicted_rounded else "INCORRECT"
    print(f"   {status} | Actual: {actual} | Predicted: {predicted_rounded} (raw: {predicted_raw:.2f})")

# Step 8: Prediction distribution
print("\nStep 8: Prediction Distribution...")

print("\nActual intensity distribution:")
print(y_test.value_counts().sort_index())

print("\nPredicted intensity distribution (rounded):")
print(pd.Series(y_pred_rounded).value_counts().sort_index())

print("\n" + "=" * 60)
print("TASK 4.2 COMPLETE: Intensity Regressor Trained")
print("=" * 60)


