# Emotional Intelligence AI System

Predict emotional states and intensity from journal reflections, then provide personalized recommendations with confidence scores.

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
5. Understand feature importance (text vs metadata)
6. Ablation study (text-only vs text+metadata)
7. Error analysis (10+ failure cases)
8. Edge deployment strategy (<100MB, offline)
9. Robustness handling (short text, missing values, contradictions)

### Success Criteria

| Criterion                | Target | Status   |
| ------------------------ | ------ | -------- |
| Emotional state accuracy | >20%   | ✅ 27.5% |
| Intensity MAE            | <2.0   | ✅ 1.280 |
| Model size               | <100MB | ✅ ~10MB |
| Offline capability       | 100%   | ✅ Yes   |

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
│   └── processed/
│       ├── X_train.pkl           # Preprocessed features
│       ├── y_train.pkl           # Targets
│       ├── feature_columns.pkl   # Column names
│       └── label_encoders.pkl    # Label encoders
│
├── src/
│   ├── __init__.py
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
│   ├── model_evaluation.py       # Evaluate both models
│   ├── model_inference.py        # Inference on new data
│   │
│   └── EDA/                      # Exploratory Data Analysis
│       ├── data_validation.py
│       ├── eda_step1.py
│       ├── eda_step2.py
│       ├── eda_step3.py
│       ├── eda_step4.py
│       └── eda_step5.py
│
├── models/
│   ├── emotion_classifier.pkl
│   ├── emotion_classifier_features.pkl
│   ├── emotion_label_encoder.pkl
│   ├── intensity_regressor.pkl
│   ├── intensity_regressor_features.pkl
│   ├── recommendation_mapping.pkl
│   ├── timing_engine.pkl
│   └── confidence_handler.pkl
│
├── outputs/
│   ├── data_validation_report.md
│   ├── emotional_state_dist.png
│   ├── intensity_dist.png
│   ├── numerical_features_dist.png
│   ├── correlation_heatmap.png
│   ├── text_length_analysis.png
│   ├── model_evaluation_report.csv
│   └── inference_results.csv
│
├── venv/                          # Virtual environment
└── tests/                         # Test scripts (future)
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
│  "I feel overwhelmed with work"                         │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Text Cleaner                                     │  │
│  │ ├── lowercase conversion                         │  │
│  │ ├── normalize whitespace                         │  │
│  │ └── char_count, word_count, is_short             │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Sentiment Analyzer (VADER)                       │  │
│  │ ├── compound score (-1 to +1)                    │  │
│  │ ├── pos, neu, neg proportions                    │  │
│  │ └── sentiment classification                     │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Text Quality Metrics                             │  │
│  │ ├── ambiguity_score (uncertain words)            │  │
│  │ ├── coherence_score (connecting words)           │  │
│  │ ├── complexity_score (unique/total)              │  │
│  │ └── emotion_word_count                           │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  Output: 12 text-derived features                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Results:**

```
Processed: 1200 text entries
Short texts (≤5 words): 232 (19.3%)
Average word count: 10.9

Sentiment Distribution:
  Positive: 496 (41.3%)
  Neutral: 392 (32.7%)
  Negative: 312 (26.0%)
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
│  ├── Normalized (0-1 scale)                            │
│  │   ├── duration_min_normalized                       │
│  │   ├── sleep_hours_normalized                        │
│  │   ├── energy_level_normalized                       │
│  │   └── stress_level_normalized                       │
│  ├── Interaction Features                              │
│  │   ├── sleep_energy_ratio                            │
│  │   ├── stress_energy_diff                            │
│  │   └── energy_stress_balance                         │
│  └── Utility Features                                  │
│      ├── sleep_hours_missing                           │
│      └── duration_per_hour                             │
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
│  │                                                  │  │
│  │ Model: XGBClassifier                             │  │
│  │ ├── n_estimators: 100                            │  │
│  │ ├── max_depth: 5                                 │  │
│  │ └── learning_rate: 0.1                           │  │
│  │                                                  │  │
│  │ Output:                                          │  │
│  │ ├── Predicted emotion (6 classes)                │  │
│  │ └── Probability distribution                     │  │
│  │     [0.05, 0.05, 0.05, 0.05, 0.75, 0.05]        │  │
│  │     └── Max = 0.75 (75% confidence)             │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Intensity Regressor (XGBoost)                    │  │
│  │                                                  │  │
│  │ Model: XGBRegressor                              │  │
│  │ ├── n_estimators: 100                            │  │
│  │ ├── max_depth: 5                                 │  │
│  │ └── learning_rate: 0.1                           │  │
│  │                                                  │  │
│  │ Output:                                          │  │
│  │ ├── Predicted intensity (1-5 scale)              │  │
│  │ └── Example: 4 (high intensity)                  │  │
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
│  Example: [0.05, 0.05, 0.05, 0.05, 0.75, 0.05]          │
│       ↓                                                 │
│  Get Base Confidence (max probability)                  │
│  Example: 0.75 (75%)                                    │
│       ↓                                                 │
│  Check Uncertainty Factors:                             │
│  ├── Is text short? → Add 15% penalty                   │
│  ├── Is text ambiguous? → Add 20% penalty               │
│  ├── Is data missing? → Add 20% penalty                 │
│  └── Is emotion rare? → Add 10% penalty                 │
│       ↓                                                 │
│  Calculate Final Confidence (base - penalty)            │
│  Example: 0.75 - 0.00 = 0.75 (75%)                      │
│       ↓                                                 │
│  Determine Uncertainty Level:                           │
│  ├── 75%+ → Low uncertainty (confident)                 │
│  ├── 50-75% → Medium uncertainty (moderate)             │
│  └── Below 50% → High uncertainty (uncertain)           │
│       ↓                                                 │
│  Output:                                                │
│  ├── Confidence score (0-100%)                          │
│  ├── Uncertainty level (low/medium/high)                │
│  ├── Message to user                                    │
│  └── Suggested action (follow/consider/evaluate)        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Confidence Calculation Examples:**

```
Scenario 1: High Confidence, Clear Text
├── Base confidence: 75%
├── Short text penalty: 0%
├── Ambiguity penalty: 0%
├── Final confidence: 75% - 0% = 75%
└── Result: "Strong recommendation" (follow)

Scenario 2: Low Confidence, Short Text
├── Base confidence: 20%
├── Short text penalty: 15%
├── Ambiguity penalty: 20%
├── Final confidence: 20% - 35% = 0% (min)
└── Result: "Use your judgment" (fallback)
```

---

### 5. Decision Engine (Day 5)

```
┌─────────────────────────────────────────────────────────┐
│              DECISION ENGINE FLOW                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: emotion + intensity + confidence                │
│  Example: overwhelmed + 4 + 75%                         │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Recommendation Mapper                            │  │
│  │                                                  │  │
│  │ Emotion → Action Mapping:                        │  │
│  │ ├── overwhelmed → Breathing exercise             │  │
│  │ ├── restless → Take a short walk                 │  │
│  │ ├── focused → Continue deep work                 │  │
│  │ ├── calm → Continue current activity             │  │
│  │ ├── mixed → Journal for 10 minutes               │  │
│  │ └── neutral → Light stretching                   │  │
│  │                                                  │  │
│  │ Intensity Adjustment:                            │  │
│  │ ├── Low (1-2): Gentle suggestions                │  │
│  │ ├── Medium (3): Standard recommendations         │  │
│  │ └── High (4-5): "IMMEDIATE:" prefix              │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Timing Engine                                    │  │
│  │                                                  │  │
│  │ Calculate Urgency (1-5):                         │  │
│  │ ├── Base urgency from emotion                    │  │
│  │ │   ├── overwhelmed: 5                           │  │
│  │ │   ├── restless: 4                              │  │
│  │ │   └── calm/focused: 1                          │  │
│  │ └── Adjust by intensity factor                   │  │
│  │                                                  │  │
│  │ Urgency Levels:                                  │  │
│  │ ├── 5: URGENT (red) - ACT NOW                    │  │
│  │ ├── 4: High Priority (orange) - Within 1 hour    │  │
│  │ ├── 3: Medium Priority (yellow) - Within 4 hrs   │  │
│  │ ├── 2: Low Priority (blue) - Within 12 hrs       │  │
│  │ └── 1: Can Wait (green) - Within 24 hrs          │  │
│  └──────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  Final Output                                           │
│  "You seem overwhelmed (75% confident).                 │
│   URGENT: Do breathing exercise within 5 minutes.       │
│   Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s."  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Completed Tasks

### Day 1: Environment Setup & EDA

#### Task 1.1-1.3: Environment Setup

- [x] Project folder structure created
- [x] Virtual environment configured
- [x] Dependencies installed via requirements.txt

#### Task 1.4: Data Validation

**Script**: `src/EDA/data_validation.py`

**Findings**:

```
Dataset: 1200 rows, 13 columns
Missing values:
  - sleep_hours: 7 (0.58%)
  - previous_day_mood: 15 (1.25%)
  - face_emotion_hint: 123 (10.25%)
Target variables:
  - emotional_state: 6 unique classes (well-balanced)
  - intensity: range [1, 5], mean=3.05
```

**Output**: `outputs/data_validation_report.md`

#### Task 1.5: Exploratory Data Analysis

| Step                         | Script               | Output                      | Key Finding                          |
| ---------------------------- | -------------------- | --------------------------- | ------------------------------------ |
| Emotional State Distribution | src/EDA/eda_step1.py | emotional_state_dist.png    | Classes well-balanced (~16-18% each) |
| Intensity Distribution       | src/EDA/eda_step2.py | intensity_dist.png          | Full 1-5 range covered               |
| Numerical Features           | src/EDA/eda_step3.py | numerical_features_dist.png | Energy/stress use 1-5 scale          |
| Correlation Heatmap          | src/EDA/eda_step4.py | correlation_heatmap.png     | Near-zero linear correlations        |
| Text Length Analysis         | src/EDA/eda_step5.py | text_length_analysis.png    | 19.3% short texts (≤5 words)         |

---

### Day 2: Text Preprocessing Module

#### Task 2.1: Text Cleaner

**Script**: `src/text_cleaner.py`

**Functionality**:

- Lowercase conversion
- Whitespace normalization
- Edge punctuation removal
- Text statistics (char_count, word_count, is_short)

**Results**:

```
Processed 1200 text entries
Short texts (≤5 words): 232 (19.3%)
Average word count: 10.9
```

#### Task 2.2: Sentiment Analyzer (VADER)

**Script**: `src/sentiment_analyzer.py`

**Functionality**:

- Compound score (-1 to +1)
- Positive/Neutral/Negative classification
- Component scores (pos, neu, neg)

**Results**:

```
Positive: 496 (41.3%)
Neutral: 392 (32.7%)
Negative: 312 (26.0%)
Average compound: 0.074 (slightly positive)
```

#### Task 2.3: Text Quality Metrics

**Script**: `src/text_quality.py`

**Metrics Calculated**:

- ambiguity_score: Proportion of uncertain words
- coherence_score: Proportion of connecting words
- complexity_score: Unique words / total words
- emotion_word_count: Count of emotion vocabulary words

**Results**:

```
Avg ambiguity: 0.016 (low uncertainty)
Avg coherence: 0.036 (stream-of-consciousness style)
Avg complexity: 0.968 (high unique word ratio)
Avg emotion words: 0.11 (users describe feelings indirectly)
```

#### Task 2.4: Full Text Pipeline

**Script**: `src/text_pipeline.py`

**Functionality**:

- Chains cleaner + sentiment + quality into single interface
- process(df) method for batch processing
- process_single_text(text) method for inference

**Output**: 12 text-derived features added to DataFrame

---

### Day 3: Feature Engineering & Fusion

#### Task 3.1: Numerical Feature Engineering

**Script**: `src/numerical_features.py`

**Features Created (9 total)**:

```
Normalized (0-1 scale):
- duration_min_normalized
- sleep_hours_normalized
- energy_level_normalized
- stress_level_normalized

Interaction Features:
- sleep_energy_ratio (sleep / energy)
- stress_energy_diff (stress - energy)
- energy_stress_balance (energy - stress)

Utility Features:
- sleep_hours_missing (indicator for missing values)
- duration_per_hour (session length in hours)
```

#### Task 3.2: Categorical Feature Encoding

**Script**: `src/categorical_features.py`

**Auto-Detection Logic**:

- Detects columns with str/object dtype AND ≤10 unique values
- Excludes text columns and targets automatically
- Handles missing values by creating 'unknown' category

**Columns Encoded (5 → 27 features)**:

```
ambience_type (5)        → ambience_type_cafe, forest, mountain, ocean, rain
time_of_day (5)          → afternoon, early_morning, evening, morning, night
previous_day_mood (6+1)  → calm, focused, mixed, neutral, overwhelmed, restless, unknown
face_emotion_hint (6+1)  → calm_face, happy_face, neutral_face, none, tense_face, tired_face, unknown
reflection_quality (3)   → clear, conflicted, vague
```

#### Task 3.3: Feature Fusion Engine

**Script**: `src/feature_fusion.py`

**Functionality**:

- Combines text + numerical + categorical features into single matrix
- Auto-detects one-hot encoded categorical columns by prefix pattern
- Separates features (X) from targets (y) for model training

**Final Feature Matrix**:

```
Feature Matrix (X)
Shape: (1200, 47)

Text Features:        11 columns
Numerical Features:    9 columns
Categorical Features: 27 columns

Targets (y)
Shape: (1200, 2)

emotional_state: 6-class label
intensity: 1-5 regression target
```

#### Task 3.4: Data Preprocessing Pipeline

**Script**: `src/preprocess_data.py`

**Functionality**:

- Runs full feature pipeline once
- Saves preprocessed data to disk (pickle format)
- Future scripts load preprocessed data (30x faster)

**Output Files**:

```
data/processed/
- X_train.pkl (preprocessed features)
- y_train.pkl (targets)
- feature_columns.pkl (column names)
- label_encoders.pkl (emotion labels)
```

---

### Day 4: Model Training

#### Task 4.1: Emotional State Classifier

**Script**: `src/emotion_classifier.py`

**Model**: XGBoost Classifier (XGBClassifier)

**Hyperparameters**:

```
n_estimators: 100
max_depth: 5
learning_rate: 0.1
min_samples_split: 10
min_samples_leaf: 5
random_state: 42
```

**Results**:

```
Overall Accuracy: 27.5%
Random Baseline: 16.7%
Improvement: +10.8 percentage points
Status: PASS (beats random baseline)
```

**Top 5 Most Important Features**:

```
1. coherence_score: 0.0416
2. ambiguity_score: 0.0323
3. char_count: 0.0323
4. emotion_word_count: 0.0321
5. pos: 0.0296
```

**Output Files**:

```
models/
- emotion_classifier.pkl
- emotion_classifier_features.pkl
- emotion_label_encoder.pkl
```

#### Task 4.2: Intensity Regressor

**Script**: `src/intensity_regressor.py`

**Model**: XGBoost Regressor (XGBRegressor)

**Hyperparameters**:

```
n_estimators: 100
max_depth: 5
learning_rate: 0.1
min_samples_split: 10
min_samples_leaf: 5
random_state: 42
```

**Results**:

```
Mean Absolute Error (MAE): 1.280
Root Mean Squared Error (RMSE): 1.525
R-squared (R2): -0.165
Exact Match Accuracy: 23.8%
Status: PASS (MAE < 2.0 target)
```

**Top 5 Most Important Features**:

```
1. previous_day_mood_restless: 0.0361
2. time_of_day_morning: 0.0361
3. face_emotion_hint_tense_face: 0.0348
4. ambience_type_cafe: 0.0343
5. previous_day_mood_overwhelmed: 0.0318
```

**Output Files**:

```
models/
- intensity_regressor.pkl
- intensity_regressor_features.pkl
```

#### Task 4.3: Model Evaluation

**Script**: `src/model_evaluation.py`

**Functionality**:

- Loads both trained models
- Evaluates on held-out test set (240 samples)
- Generates summary report

**Combined Results**:

```
Both predictions correct: 13.8%
Classifier only correct: 15.0%
Regressor only correct: 29.2%
Both incorrect: 42.1%
```

**Output Files**:

```
outputs/
- model_evaluation_report.csv
```

#### Task 4.4: Model Inference

**Script**: `src/model_inference.py`

**Functionality**:

- Demonstrates inference on new data
- Shows sample predictions with confidence scores
- Saves inference results

**Output Files**:

```
outputs/
- inference_results.csv
```

---

### Day 5: Decision Engine

#### Task 5.1: Recommendation Mapper

**Script**: `src/recommendation_mapper.py`

**Functionality**:

- Maps 6 emotions to specific actions
- Adjusts recommendations based on intensity
- Saves mapping to pickle file

**Recommendations**:

```
| Emotion     | Action                 | Category | Duration     |
|-------------|------------------------|----------|--------------|
| calm        | Continue activity      | maintain | N/A          |
| focused     | Deep work session      | maintain | 25 minutes   |
| mixed       | Journal                | reflect  | 10 minutes   |
| neutral     | Light stretching       | activate | 5-10 minutes |
| overwhelmed | Breathing exercise     | calm     | 5 minutes    |
| restless    | Take a short walk      | activate | 10-15 minutes|
```

**Output Files**:

```
models/
- recommendation_mapping.pkl
```

#### Task 5.2: Timing Engine

**Script**: `src/timing_engine.py`

**Functionality**:

- Calculates urgency score (1-5)
- Determines timeframe for action
- Considers time of day modifiers

**Urgency Levels**:

```
| Level | Label         | Color  | Timeframe      |
|-------|---------------|--------|----------------|
| 5     | URGENT        | red    | ACT NOW        |
| 4     | High Priority | orange | Within 1 hour  |
| 3     | Medium        | yellow | Within 4 hours |
| 2     | Low Priority  | blue   | Within 12 hours|
| 1     | Can Wait      | green  | Within 24 hours|
```

**Output Files**:

```
models/
- timing_engine.pkl
```

#### Task 5.3: Confidence Handler

**Script**: `src/confidence_handler.py`

**Functionality**:

- Calculates prediction confidence
- Applies uncertainty penalties
- Determines fallback recommendations

**Uncertainty Penalties**:

```
| Factor              | Penalty |
|---------------------|---------|
| Short text (≤5 words) | 15%   |
| High ambiguity      | 20%     |
| Missing data        | 20%     |
| Rare emotion        | 10%     |
```

**Output Files**:

```
models/
- confidence_handler.pkl
```

---

## Key Insights

### Data Quality

| Aspect          | Status     | Notes                                       |
| --------------- | ---------- | ------------------------------------------- |
| Class Balance   | Excellent  | All 6 emotional states ~16-18%              |
| Intensity Range | Good       | Full 1-5 scale covered                      |
| Missing Values  | Managed    | Handled via 'unknown' category + indicators |
| Feature Scales  | Normalized | All numerical features on 0-1 scale         |

### Model Performance

| Model      | Metric       | Result    | Target | Status        |
| ---------- | ------------ | --------- | ------ | ------------- |
| Classifier | Accuracy     | 27.5%     | >20%   | ✅ PASS       |
| Classifier | Baseline     | 16.7%     | -      | -             |
| Classifier | Improvement  | +10.8 pts | -      | ✅ PASS       |
| Regressor  | MAE          | 1.280     | <2.0   | ✅ PASS       |
| Regressor  | R-squared    | -0.165    | >0     | ⚠️ NEEDS WORK |
| Regressor  | Exact Match  | 23.8%     | -      | ✅ ACCEPTABLE |
| Combined   | Both Correct | 13.8%     | -      | ✅ ACCEPTABLE |

### System Capabilities

| Component              | Status      | Output                     |
| ---------------------- | ----------- | -------------------------- |
| Text Processing        | ✅ Complete | 12 text features           |
| Feature Engineering    | ✅ Complete | 47 total features          |
| Emotion Prediction     | ✅ Complete | 6-class classification     |
| Intensity Prediction   | ✅ Complete | 1-5 regression             |
| Recommendation Mapping | ✅ Complete | 6 emotion-action pairs     |
| Timing/Urgency         | ✅ Complete | 5 urgency levels           |
| Confidence Handling    | ✅ Complete | Uncertainty quantification |

---

## Key Calculations

### Confidence Score Calculation

```
Formula:
├── Final Confidence = Base Confidence - Uncertainty Penalty
│
├── Base Confidence = max(probability_distribution)
│   └── Example: max([0.05, 0.05, 0.05, 0.05, 0.75, 0.05]) = 0.75
│
└── Uncertainty Penalty = sum(all applicable penalties)
    ├── Short text (≤5 words): +0.15
    ├── High ambiguity (>0.05): +0.20
    ├── Missing data: +0.20
    └── Rare emotion: +0.10

Example Calculation:
├── Input: "ok" (short text)
├── Model prediction: [0.20, 0.20, 0.20, 0.20, 0.10, 0.10]
├── Base confidence: 0.20 (20%)
├── Penalties:
│   ├── Short text: 0.15
│   └── Total: 0.15
└── Final confidence: 0.20 - 0.15 = 0.05 (5%)
    └── Result: High uncertainty → Use fallback
```

### Urgency Score Calculation

```
Formula:
├── Urgency Score = Base Urgency × Intensity Factor
│
├── Base Urgency (by emotion):
│   ├── overwhelmed: 5
│   ├── restless: 4
│   ├── mixed: 3
│   ├── neutral: 2
│   └── calm/focused: 1
│
└── Intensity Factor = intensity / 3.0
    └── Example: intensity 4 → 4/3 = 1.33

Example Calculation:
├── Input: overwhelmed + intensity 5
├── Base urgency: 5
├── Intensity factor: 5/3 = 1.67
├── Urgency score: 5 × 1.67 = 8.35
├── Clamp to [1, 5]: 5
└── Result: URGENT (ACT NOW)
```

---

## How to Run Scripts

### Prerequisites

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify dependencies
pip list
```

### Run Full Pipeline

```powershell
# Preprocess data (run once)
python src/preprocess_data.py

# Train emotion classifier
python src/emotion_classifier.py

# Train intensity regressor
python src/intensity_regressor.py

# Evaluate both models
python src/model_evaluation.py

# Test inference
python src/model_inference.py
```

### View Outputs

```powershell
# Markdown reports
code outputs/data_validation_report.md

# PNG visualizations
Start-Process outputs/emotional_state_dist.png

# CSV reports
code outputs/model_evaluation_report.csv
```

---

## Dependencies

Core packages (see requirements.txt):

```
pandas==2.0.3          # Data manipulation
numpy==1.24.3          # Numerical operations
scikit-learn==1.3.0    # Machine learning utilities
xgboost==2.0.3         # Model training
matplotlib==3.7.2      # Basic plotting
seaborn==0.12.2        # Statistical visualizations
vaderSentiment==3.3.2  # Text sentiment analysis
```

Install with:

```powershell
pip install -r requirements.txt
```

---

## Model Files

### Trained Models

| File                             | Size | Purpose                 |
| -------------------------------- | ---- | ----------------------- |
| emotion_classifier.pkl           | ~5MB | Predict emotional state |
| emotion_classifier_features.pkl  | <1KB | Feature column names    |
| emotion_label_encoder.pkl        | <1KB | Decode predictions      |
| intensity_regressor.pkl          | ~5MB | Predict intensity       |
| intensity_regressor_features.pkl | <1KB | Feature column names    |
| recommendation_mapping.pkl       | <1KB | Emotion→Action          |
| timing_engine.pkl                | <1KB | Urgency logic           |
| confidence_handler.pkl           | <1KB | Confidence scores       |

**Total Model Size**: ~10MB (well under 100MB edge deployment target)

---

## Next Steps

### Day 6-7: Feedback Collection

- [ ] Build feedback collection system
- [ ] Design feedback database schema
- [ ] Add user rating interface

### Day 8-10: Personalization

- [ ] Analyze collected feedback
- [ ] Train personalized recommendation model
- [ ] Compare learned vs hardcoded performance

### Day 11+: Advanced Features

- [ ] TabNet model comparison
- [ ] Multi-user personalization
- [ ] A/B testing framework
- [ ] Edge deployment packaging

---

## Contributing

1. Follow the one-task-at-a-time approach
2. Document all findings in this README or separate reports
3. Test scripts before marking tasks complete
4. Share outputs for review before proceeding

---

## Notes

- ✅ All processing designed for local/offline execution
- ✅ Model size target: <100MB total for edge deployment
- ✅ Privacy: No data leaves the device in final implementation
- ✅ Uncertainty quantification is central to decision logic
- ✅ Feature matrix: 47 features ready for model training

---

**Last Updated:** Day 5 Complete - Decision Engine  
**Next:** Day 6 - Feedback Collection System  
**Status:** All core components operational

```

```
