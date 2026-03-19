# src/emotion_classifier.py
"""
Day 4 - Task 4.1: Emotional State Classifier
Purpose: Train a model to predict emotional state from journal entries
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

print("=" * 60)
print("DAY 4 - TASK 4.1: EMOTIONAL STATE CLASSIFIER")
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

# Get only emotional_state for classification
y_emotion = y['emotional_state']

# Encode labels for XGBoost (requires numeric labels)
label_encoder = LabelEncoder()
y_emotion_encoded = label_encoder.fit_transform(y_emotion)

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y_emotion.shape}")
print(f"Classes: {label_encoder.classes_}")

# Step 2: Split data
print("\nStep 2: Splitting data into train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_emotion_encoded, 
    test_size=0.2,
    random_state=42,
    stratify=y_emotion_encoded
)

print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

# Step 3: Train classifier
print("\nStep 3: Training XGBoost Classifier...")

classifier = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

print("   Training... (this may take 30-60 seconds)")
classifier.fit(X_train, y_train)
print("   Training complete!")

# Step 4: Evaluate on test set
print("\nStep 4: Evaluating model performance...")

y_pred_encoded = classifier.predict(X_test)

# Decode predictions back to string labels
y_pred = label_encoder.inverse_transform(y_pred_encoded)
y_test_labels = label_encoder.inverse_transform(y_test)

# Overall accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
baseline = 1 / len(label_encoder.classes_)

print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Random Baseline: {baseline:.3f} ({baseline*100:.1f}%)")
print(f"Improvement: +{(accuracy - baseline)*100:.1f} percentage points")

if accuracy > baseline:
    print("   Model beats random baseline!")
else:
    print("   Model needs improvement")

# Detailed classification report
print(f"\nClassification Report:")
print(classification_report(y_test_labels, y_pred))

# Confusion matrix
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test_labels, y_pred, labels=label_encoder.classes_)
print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))

# Step 5: Feature Importance
print("\nStep 5: Top 10 Most Important Features...")

importances = classifier.feature_importances_
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

with open('models/emotion_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('models/emotion_classifier_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

with open('models/emotion_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("   Model saved: models/emotion_classifier.pkl")
print("   Features saved: models/emotion_classifier_features.pkl")
print("   Label encoder saved: models/emotion_label_encoder.pkl")

# Step 7: Test on sample predictions
print("\nStep 7: Sample Predictions...")

sample_indices = X_test.index[:5]
for idx in sample_indices:
    actual = y_test_labels[list(X_test.index).index(idx)]
    predicted_idx = list(X_test.index).index(idx)
    predicted = y_pred[predicted_idx]
    proba = classifier.predict_proba(X_test)[predicted_idx].max()
    
    status = "CORRECT" if actual == predicted else "INCORRECT"
    print(f"   {status} | Actual: {actual:12} | Predicted: {predicted:12} | Confidence: {proba:.2f}")

print("\n" + "=" * 60)
print("TASK 4.1 COMPLETE: Emotional State Classifier Trained")
print("=" * 60)