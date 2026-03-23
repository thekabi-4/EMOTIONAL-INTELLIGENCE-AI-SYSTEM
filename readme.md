# Emotional Intelligence AI System

Predict emotional states and intensity from journal reflections, then provide personalized recommendations with confidence scores and feedback collection.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              EMOTIONAL INTELLIGENCE AI SYSTEM           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Journal Entry → Text Processing → Features → Models   │
│                                           ↓             │
│  Recommendation ← Decision Engine ← Confidence         │
│       ↓                                                 │
│  User Feedback → Feedback Database → Personalization   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Project Goals

### Core Objectives

1. Predict emotional state (6 classes: calm, focused, restless, neutral, overwhelmed, mixed)
2. Predict intensity level (regression, 1-5 scale)
3. Recommend action + timing based on predictions
4. Quantify prediction uncertainty (confidence score + uncertainty flags)
5. Collect user feedback for continuous improvement
6. Personalize recommendations based on user history
7. Edge deployment strategy (<100MB, offline)

### Success Criteria

| Criterion                | Target | Status         |
| ------------------------ | ------ | -------------- |
| Emotional state accuracy | >20%   | ✅ 27.5%       |
| Intensity MAE            | <2.0   | ✅ 1.280       |
| Model size               | <100MB | ✅ ~10MB       |
| Offline capability       | 100%   | ✅ Yes         |
| Feedback collection      | Yes    | ✅ Implemented |

---

## Project Structure

```
Emotional_Intelligence_AI_System/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── train_data.csv            # Training data (1200 rows, labeled)
│   ├── test_data.csv             # Test data (120 rows, unlabeled)
│   ├── processed/
│   │   ├── X_train.pkl           # Preprocessed features
│   │   ├── y_train.pkl           # Targets
│   │   ├── feature_columns.pkl   # Column names
│   │   └── label_encoders.pkl    # Label encoders
│   └── feedback/
│       └── feedback_YYYY-MM-DD.jsonl  # Daily feedback logs
│
├── src/
│   ├── text_cleaner.py           # Text cleaning and normalization
│   ├── sentiment_analyzer.py     # VADER sentiment analysis
│   ├── text_quality.py           # Text quality metrics
│   ├── text_pipeline.py          # Full text processing pipeline
│   ├── numerical_features.py     # Numerical feature engineering
│   ├── categorical_features.py   # Categorical encoding
│   ├── feature_fusion.py         # Combine all features
│   ├── preprocess_data.py        # Preprocess and save data
│   ├── emotion_classifier.py     # XGBoost classifier
│   ├── intensity_regressor.py    # XGBoost regressor
│   ├── recommendation_mapper.py  # Emotion to action mapping
│   ├── timing_engine.py          # Urgency determination
│   ├── confidence_handler.py     # Uncertainty flags
│   ├── feedback_schema.py        # Feedback data structure
│   ├── feedback_recorder.py      # Log feedback to files
│   ├── feedback_database.py      # Query & analyze feedback
│   ├── decision_pipeline.py      # End-to-end integration
│   ├── model_evaluation.py       # Evaluate both models
│   └── model_inference.py        # Inference on new data
│
├── models/
│   ├── emotion_classifier.pkl
│   ├── emotion_label_encoder.pkl
│   ├── intensity_regressor.pkl
│   ├── recommendation_mapping.pkl
│   ├── timing_engine.pkl
│   └── confidence_handler.pkl
│
├── outputs/
│   ├── model_evaluation_report.csv
│   ├── feedback_schema.json
│   └── feedback_analysis.csv
│
└── venv/                          # Virtual environment
```

---

## Key Workflows

### 1. Text Processing Pipeline (Day 2)

```
┌─────────────────────────────────────────────────────────┐
│              TEXT PROCESSING FLOW                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Raw Journal Text                                       │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Text Cleaner                                     │  │
│  │ ├── lowercase, normalize whitespace              │  │
│  │ └── char_count, word_count, is_short             │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Sentiment Analyzer (VADER)                       │  │
│  │ ├── compound, pos, neu, neg                      │  │
│  │ └── sentiment classification                     │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Text Quality Metrics                             │  │
│  │ ├── ambiguity_score, coherence_score             │  │
│  │ ├── complexity_score, emotion_word_count         │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  Output: 12 text-derived features                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 2. Feature Engineering (Day 3)

```
┌─────────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING FLOW                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Text Features (11)                                     │
│  ├── char_count, word_count, is_short                  │
│  ├── compound, pos, neu, neg                           │
│  └── ambiguity, coherence, complexity, emotion_words   │
│                                                         │
│  Numerical Features (9)                                 │
│  ├── Normalized: duration, sleep, energy, stress       │
│  ├── Interactions: sleep_energy_ratio, etc.            │
│  └── Utility: sleep_hours_missing, duration_per_hour   │
│                                                         │
│  Categorical Features (27)                              │
│  ├── ambience_type (5 one-hot encoded)                 │
│  ├── time_of_day (5 one-hot encoded)                   │
│  ├── previous_day_mood (7 one-hot encoded)             │
│  ├── face_emotion_hint (7 one-hot encoded)             │
│  └── reflection_quality (3 one-hot encoded)            │
│                                                         │
│       ↓                                                 │
│  Feature Fusion                                         │
│  ├── Total: 47 features                                 │
│  ├── X matrix: (1200, 47)                              │
│  └── y targets: (1200, 2)                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 3. Model Prediction (Day 4)

```
┌─────────────────────────────────────────────────────────┐
│              MODEL PREDICTION FLOW                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: 47 features                                     │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Emotion Classifier (XGBoost)                     │  │
│  │ ├── Output: 6-class emotion                      │  │
│  │ └── Confidence: probability distribution         │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Intensity Regressor (XGBoost)                    │  │
│  │ ├── Output: intensity (1-5 scale)                │  │
│  │ └── MAE: 1.280                                   │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Performance:                                           │
│  ├── Classifier Accuracy: 27.5% (target: >20%) ✅      │
│  └── Regressor MAE: 1.280 (target: <2.0) ✅            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 4. Confidence Handler (Day 5)

```
┌─────────────────────────────────────────────────────────┐
│            CONFIDENCE HANDLER FLOW                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Model Prediction (probabilities)                       │
│       ↓                                                 │
│  Get Base Confidence (max probability)                  │
│       ↓                                                 │
│  Check Uncertainty Factors:                             │
│  ├── Is text short? → Add 15% penalty                   │
│  ├── Is text ambiguous? → Add 20% penalty               │
│  ├── Is data missing? → Add 20% penalty                 │
│  └── Is emotion rare? → Add 10% penalty                 │
│       ↓                                                 │
│  Calculate Final Confidence (base - penalty)            │
│       ↓                                                 │
│  Determine Uncertainty Level:                           │
│  ├── 75%+ → Low uncertainty (confident)                 │
│  ├── 50-75% → Medium uncertainty (moderate)             │
│  └── Below 50% → High uncertainty (uncertain)           │
│       ↓                                                 │
│  Output: confidence score + message + action            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 5. Decision Engine (Day 5)

```
┌─────────────────────────────────────────────────────────┐
│              DECISION ENGINE FLOW                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: emotion + intensity + confidence                │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Recommendation Mapper                            │  │
│  │ ├── overwhelmed → Breathing exercise             │  │
│  │ ├── restless → Take a short walk                 │  │
│  │ ├── focused → Continue deep work                 │  │
│  │ ├── calm → Continue current activity             │  │
│  │ ├── mixed → Journal for 10 minutes               │  │
│  │ └── neutral → Light stretching                   │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Timing Engine                                    │  │
│  │ ├── Urgency 5: URGENT (ACT NOW)                  │  │
│  │ ├── Urgency 4: High Priority (1 hour)            │  │
│  │ ├── Urgency 3: Medium (4 hours)                  │  │
│  │ ├── Urgency 2: Low Priority (12 hours)           │  │
│  │ └── Urgency 1: Can Wait (24 hours)               │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  Final Recommendation to User                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 6. Feedback Collection (Day 6)

```
┌─────────────────────────────────────────────────────────┐
│              FEEDBACK COLLECTION FLOW                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User receives recommendation                           │
│       ↓                                                 │
│  User follows recommendation? (Yes/No)                  │
│       ↓                                                 │
│  User rates outcome (1-10)                              │
│       ↓                                                 │
│  FeedbackRecorder.log_feedback()                        │
│       ↓                                                 │
│  Save to: data/feedback/feedback_YYYY-MM-DD.jsonl       │
│       ↓                                                 │
│  Later: Analyze → Personalize recommendations           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Feedback Record Structure:**

```json
{
  "feedback_id": "fb_xxxxxxxx",
  "timestamp": "2026-03-23T12:00:00",
  "user_id": "user_001",
  "emotion_predicted": "overwhelmed",
  "emotion_confidence": 0.75,
  "intensity_predicted": 4,
  "recommendation": "breathing_exercise",
  "urgency_level": 5,
  "user_followed": true,
  "outcome_rating": 8,
  "user_notes": "Helped calm down quickly"
}
```

---

### 7. Full Decision Pipeline (Day 6)

```
┌─────────────────────────────────────────────────────────┐
│              FULL PIPELINE FLOW                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. pipeline.predict(journal_text, metadata)            │
│       ↓                                                 │
│  2. Returns: emotion + intensity + recommendation       │
│       ↓                                                 │
│  3. User sees recommendation + provides feedback        │
│       ↓                                                 │
│  4. pipeline.log_feedback(response, user_id, ...)       │
│       ↓                                                 │
│  5. Feedback auto-saved to data/feedback/               │
│       ↓                                                 │
│  6. Later: Analyze feedback → Improve recommendations   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Completed Tasks

### Day 1: Environment Setup & EDA

| Task                       | Status      | Output                    |
| -------------------------- | ----------- | ------------------------- |
| 1.1-1.3: Environment Setup | ✅ Complete | venv, requirements.txt    |
| 1.4: Data Validation       | ✅ Complete | data_validation_report.md |
| 1.5: EDA (5 steps)         | ✅ Complete | 5 PNG visualizations      |

---

### Day 2: Text Preprocessing Module

| Task                      | Status      | Output                |
| ------------------------- | ----------- | --------------------- |
| 2.1: Text Cleaner         | ✅ Complete | text_cleaner.py       |
| 2.2: Sentiment Analyzer   | ✅ Complete | sentiment_analyzer.py |
| 2.3: Text Quality Metrics | ✅ Complete | text_quality.py       |
| 2.4: Full Text Pipeline   | ✅ Complete | text_pipeline.py      |

**Output:** 12 text-derived features

---

### Day 3: Feature Engineering & Fusion

| Task                      | Status      | Output             |
| ------------------------- | ----------- | ------------------ |
| 3.1: Numerical Features   | ✅ Complete | 9 features         |
| 3.2: Categorical Encoding | ✅ Complete | 27 features        |
| 3.3: Feature Fusion       | ✅ Complete | 47 total features  |
| 3.4: Preprocess Pipeline  | ✅ Complete | 30x faster loading |

**Output:** X (1200, 47), y (1200, 2)

---

### Day 4: Model Training

| Task                     | Status      | Output                  |
| ------------------------ | ----------- | ----------------------- |
| 4.1: Emotion Classifier  | ✅ Complete | XGBoost, 27.5% accuracy |
| 4.2: Intensity Regressor | ✅ Complete | XGBoost, MAE 1.280      |
| 4.3: Model Evaluation    | ✅ Complete | evaluation_report.csv   |
| 4.4: Model Inference     | ✅ Complete | inference_results.csv   |

**Performance:**

```
Classifier: 27.5% accuracy (baseline: 16.7%) ✅
Regressor:  MAE 1.280 (target: <2.0) ✅
Combined:   13.8% both correct ✅
```

---

### Day 5: Decision Engine

| Task                       | Status      | Output                     |
| -------------------------- | ----------- | -------------------------- |
| 5.1: Recommendation Mapper | ✅ Complete | 6 emotion-action mappings  |
| 5.2: Timing Engine         | ✅ Complete | 5 urgency levels           |
| 5.3: Confidence Handler    | ✅ Complete | Uncertainty quantification |

**Recommendations:**

```
| Emotion     | Action                 | Duration     |
|-------------|------------------------|--------------|
| calm        | Continue activity      | N/A          |
| focused     | Deep work session      | 25 minutes   |
| mixed       | Journal                | 10 minutes   |
| neutral     | Light stretching       | 5-10 minutes |
| overwhelmed | Breathing exercise     | 5 minutes    |
| restless    | Take a short walk      | 10-15 minutes|
```

---

### Day 6: Feedback Collection System

| Task                      | Status      | Output               |
| ------------------------- | ----------- | -------------------- |
| 6.1: Feedback Schema      | ✅ Complete | feedback_schema.py   |
| 6.2: Feedback Recorder    | ✅ Complete | feedback_recorder.py |
| 6.3: Feedback Database    | ✅ Complete | feedback_database.py |
| 6.4: Pipeline Integration | ✅ Complete | decision_pipeline.py |

**Feedback Capabilities:**

```
✅ Log user feedback (followed + rating)
✅ Store in JSONL format (daily files)
✅ Query by user, date, emotion, recommendation
✅ Analyze recommendation effectiveness
✅ Export to CSV for external analysis
✅ Auto-save from decision pipeline
```

---

## Model Performance Summary

```
┌─────────────────────────────────────────────────────────┐
│              MODEL PERFORMANCE SUMMARY                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Emotion Classifier (XGBoost)                           │
│  ├── Accuracy: 27.5%                                    │
│  ├── Baseline: 16.7%                                    │
│  ├── Improvement: +10.8 points ✅                       │
│  └── Top Features:                                      │
│      ├── coherence_score: 0.0416                        │
│      ├── ambiguity_score: 0.0323                        │
│      └── char_count: 0.0323                             │
│                                                         │
│  Intensity Regressor (XGBoost)                          │
│  ├── MAE: 1.280 (target: <2.0) ✅                       │
│  ├── RMSE: 1.525                                        │
│  ├── R-squared: -0.165                                  │
│  └── Exact Match: 23.8%                                 │
│                                                         │
│  Combined Performance                                   │
│  ├── Both predictions correct: 13.8%                    │
│  ├── Classifier only: 15.0%                             │
│  ├── Regressor only: 29.2%                              │
│  └── Both incorrect: 42.1%                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Key Calculations

### Confidence Score

```
Final Confidence = Base Confidence - Uncertainty Penalty

Base Confidence = max(probability_distribution)

Uncertainty Penalty:
├── Short text (≤5 words): +0.15
├── High ambiguity (>0.05): +0.20
├── Missing data: +0.20
└── Rare emotion: +0.10

Example:
├── Base: 75%
├── Penalties: 0%
└── Final: 75% → Low uncertainty (confident)
```

### Urgency Score

```
Urgency Score = Base Urgency × Intensity Factor

Base Urgency by Emotion:
├── overwhelmed: 5
├── restless: 4
├── mixed: 3
├── neutral: 2
└── calm/focused: 1

Intensity Factor = intensity / 3.0

Example:
├── overwhelmed + intensity 5
├── Base: 5 × (5/3) = 8.35
├── Clamp to [1, 5]: 5
└── Result: URGENT (ACT NOW)
```

---

## How to Run

### Quick Start

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Preprocess data (once)
python src/preprocess_data.py

# 3. Train models
python src/emotion_classifier.py
python src/intensity_regressor.py

# 4. Evaluate
python src/model_evaluation.py

# 5. Test full pipeline
python src/decision_pipeline.py
```

### View Results

```powershell
# View feedback data
code data/feedback/feedback_*.jsonl

# View analysis
code outputs/feedback_analysis.csv

# View model evaluation
code outputs/model_evaluation_report.csv
```

---

## Dependencies

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
vaderSentiment==3.3.2
```

**Install:**

```powershell
pip install -r requirements.txt
```

---

## Model Files

| File                       | Size | Purpose            |
| -------------------------- | ---- | ------------------ |
| emotion_classifier.pkl     | ~5MB | Predict emotion    |
| emotion_label_encoder.pkl  | <1KB | Decode predictions |
| intensity_regressor.pkl    | ~5MB | Predict intensity  |
| recommendation_mapping.pkl | <1KB | Emotion→Action     |
| timing_engine.pkl          | <1KB | Urgency logic      |
| confidence_handler.pkl     | <1KB | Confidence scores  |

**Total:** ~10MB (well under 100MB target)

---

## Next Steps

### Day 7: Personalization Engine

- [ ] Task 7.1: User Profile Builder
- [ ] Task 7.2: Recommendation Ranker
- [ ] Task 7.3: Adaptive Timing
- [ ] Task 7.4: Personalization Pipeline

### Day 8-10: Advanced Features

- [ ] Multi-user support
- [ ] A/B testing framework
- [ ] Model comparison (TabNet vs XGBoost)
- [ ] Edge deployment packaging

### Day 11+: Production Readiness

- [ ] API wrapper
- [ ] Mobile integration
- [ ] Privacy enhancements
- [ ] Performance optimization

---

## Contributing

1. Follow the one-task-at-a-time approach
2. Document all findings in this README
3. Test scripts before marking complete
4. Share outputs for review

---

## Notes

- ✅ All processing: local/offline
- ✅ Model size: <100MB
- ✅ Privacy: No data leaves device
- ✅ Uncertainty: Quantified
- ✅ Features: 47 total
- ✅ Feedback: Auto-collected

---

**Last Updated:** Day 6 Complete - Feedback Collection System  
**Next:** Day 7 - Personalization Engine  
**Status:** Full pipeline operational with feedback loop

```

```
