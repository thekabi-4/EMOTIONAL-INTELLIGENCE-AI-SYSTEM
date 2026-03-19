# Emotional Intelligence AI System

> Predict emotional states and intensity from journal reflections, then provide personalized recommendations.

## 📋 Project Status

**Current Phase**: Day 3 - Feature Engineering & Fusion ✅ Complete

**Next Phase**: Day 4 - Model Training

**Last Updated**: $(Get-Date -Format "yyyy-MM-dd")

---

## 🎯 Project Goals

### Core Objectives

1. **Part 1**: Predict emotional state (6 classes: calm, focused, restless, neutral, overwhelmed, mixed)
2. **Part 2**: Predict intensity level (regression, 1-5 scale)
3. **Part 3**: Recommend action + timing based on predictions
4. **Part 4**: Quantify prediction uncertainty (confidence score + uncertain flag)
5. **Part 5**: Understand feature importance (text vs metadata)
6. **Part 6**: Ablation study (text-only vs text+metadata)
7. **Part 7**: Error analysis (10+ failure cases)
8. **Part 8**: Edge deployment strategy (<100MB, offline)
9. **Part 9**: Robustness handling (short text, missing values, contradictions)

### Success Criteria

- [ ] Emotional state accuracy > 20% (vs 16.7% random baseline)
- [ ] Intensity MAE < 2.0
- [ ] Model size < 100MB for edge deployment
- [ ] 100% offline capability
- [ ] All 9 parts documented and validated

---

## 📁 Project Structure

```
Emotional_Intelligence_AI_System/
│
├── 📄 README.md                      # This file
├── 📄 requirements.txt               # Python dependencies
│
├── 📁 data/
│   ├── 📄 train_data.csv            # Training data (1200 rows, labeled)
│   └──  test_data.csv             # Test data (120 rows, unlabeled)
│
├── 📁 src/
│   ├── 📄 data_validation.py        # Day 1: Validate dataset structure
│   ├── 📄 eda_step1.py              # Day 1: Emotional state distribution
│   ├── 📄 eda_step2.py              # Day 1: Intensity distribution
│   ├── 📄 eda_step3.py              # Day 1: Numerical features histograms
│   ├── 📄 eda_step4.py              # Day 1: Correlation heatmap
│   ├── 📄 eda_step5.py              # Day 1: Text length analysis
│   ├── 📄 text_cleaner.py           # Day 2: Clean and normalize text
│   ├── 📄 sentiment_analyzer.py     # Day 2: VADER sentiment analysis
│   ├── 📄 text_quality.py           # Day 2: Text quality metrics
│   ├── 📄 text_pipeline.py          # Day 2: Full text processing pipeline
│   ├── 📄 numerical_features.py     # Day 3: Numerical feature engineering
│   ├── 📄 categorical_features.py   # Day 3: Categorical encoding
│   └──  feature_fusion.py         # Day 3: Combine all features
│
├── 📁 outputs/
│   ├── 📄 data_validation_report.md # Validation results
│   ├── 📄 emotional_state_dist.png  # EDA visualization
│   ├── 📄 intensity_dist.png        # EDA visualization
│   ├── 📄 numerical_features_dist.png # EDA visualization
│   ├── 📄 correlation_heatmap.png   # EDA visualization
│   └──  text_length_analysis.png  # EDA visualization
│
├──  models/                       # (Future) Trained model files
├── 📁 tests/                        # (Future) Test scripts
└──  analysis/                     # (Future) Error analysis reports
```

---

## 🚀 Completed Tasks

### ✅ Day 1: Environment Setup & EDA

#### Task 1.1-1.3: Environment Setup

- [x] Project folder structure created
- [x] Virtual environment configured
- [x] Dependencies installed via `requirements.txt`

#### Task 1.4: Data Validation

**Script**: `src/data_validation.py`

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

| Step                         | Script             | Output                        | Key Finding                                            |
| ---------------------------- | ------------------ | ----------------------------- | ------------------------------------------------------ |
| Emotional State Distribution | `src/eda_step1.py` | `emotional_state_dist.png`    | Classes well-balanced (~16-18% each)                   |
| Intensity Distribution       | `src/eda_step2.py` | `intensity_dist.png`          | Full 1-5 range covered                                 |
| Numerical Features           | `src/eda_step3.py` | `numerical_features_dist.png` | Energy/stress use 1-5 scale                            |
| Correlation Heatmap          | `src/eda_step4.py` | `correlation_heatmap.png`     | Near-zero linear correlations → need non-linear models |
| Text Length Analysis         | `src/eda_step5.py` | `text_length_analysis.png`    | 19.3% short texts (≤5 words)                           |

---

### ✅ Day 2: Text Preprocessing Module

#### Task 2.1: Text Cleaner

**Script**: `src/text_cleaner.py`

**Functionality**:

- Lowercase conversion
- Whitespace normalization
- Edge punctuation removal
- Text statistics (char_count, word_count, is_short)

**Results**:

```
✅ Processed 1200 text entries
✅ Short texts (≤5 words): 232 (19.3%)
✅ Average word count: 10.9
```

#### Task 2.2: Sentiment Analyzer (VADER)

**Script**: `src/sentiment_analyzer.py`

**Functionality**:

- Compound score (-1 to +1)
- Positive/Neutral/Negative classification
- Component scores (pos, neu, neg)

**Results**:

```
✅ Positive: 496 (41.3%)
✅ Neutral: 392 (32.7%)
✅ Negative: 312 (26.0%)
✅ Average compound: 0.074 (slightly positive)
```

#### Task 2.3: Text Quality Metrics

**Script**: `src/text_quality.py`

**Metrics Calculated**:

- `ambiguity_score`: Proportion of uncertain words (maybe, perhaps, etc.)
- `coherence_score`: Proportion of connecting words (and, but, because)
- `complexity_score`: Unique words / total words
- `emotion_word_count`: Count of emotion vocabulary words

**Results**:

```
✅ Avg ambiguity: 0.016 (low uncertainty)
✅ Avg coherence: 0.036 (stream-of-consciousness style)
✅ Avg complexity: 0.968 (high unique word ratio - short texts)
✅ Avg emotion words: 0.11 (users describe feelings indirectly)
```

#### Task 2.4: Full Text Pipeline

**Script**: `src/text_pipeline.py`

**Functionality**:

- Chains cleaner + sentiment + quality into single interface
- `process(df)` method for batch processing
- `process_single_text(text)` method for inference

**Output**: 13 text-derived features added to DataFrame

---

### ✅ Day 3: Feature Engineering & Fusion

#### Task 3.1: Numerical Feature Engineering

**Script**: `src/numerical_features.py`

**Features Created (9 total)**:

```
Normalized (0-1 scale):
├── duration_min_normalized
├── sleep_hours_normalized
├── energy_level_normalized
└── stress_level_normalized

Interaction Features:
├── sleep_energy_ratio (sleep ÷ energy)
├── stress_energy_diff (stress - energy)
└── energy_stress_balance (energy - stress)

Utility Features:
├── sleep_hours_missing (indicator for missing values)
└── duration_per_hour (session length in hours)
```

**Key Insight**: Normalization ensures all numerical features are on comparable 0-1 scale for model training.

#### Task 3.2: Categorical Feature Encoding

**Script**: `src/categorical_features.py`

**Auto-Detection Logic**:

- Detects columns with `str`/`object` dtype AND ≤10 unique values
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

**Key Insight**: Missing values preserved as 'unknown' category (informative for uncertainty modeling).

#### Task 3.3: Feature Fusion Engine

**Script**: `src/feature_fusion.py`

**Functionality**:

- Combines text + numerical + categorical features into single matrix
- Auto-detects one-hot encoded categorical columns by prefix pattern
- Separates features (X) from targets (y) for model training

**Final Feature Matrix**:

```
┌─────────────────────────────────────┐
│ Feature Matrix (X)                  │
│ Shape: (1200, 48)                   │
├─────────────────────────────────────┤
│ Text Features:        12 columns    │
│ Numerical Features:    9 columns    │
│ Categorical Features: 27 columns    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Targets (y)                         │
│ Shape: (1200, 2)                    │
├─────────────────────────────────────┤
│ emotional_state: 6-class label      │
│ intensity: 1-5 regression target    │
└─────────────────────────────────────┘
```

---

## 🔑 Key Insights from Days 1-3

### Data Quality

| Aspect          | Status        | Notes                                       |
| --------------- | ------------- | ------------------------------------------- |
| Class Balance   | ✅ Excellent  | All 6 emotional states ~16-18%              |
| Intensity Range | ✅ Good       | Full 1-5 scale covered                      |
| Missing Values  | ⚠️ Managed    | Handled via 'unknown' category + indicators |
| Feature Scales  | ✅ Normalized | All numerical features on 0-1 scale         |

### Modeling Strategy

1. **Classification**: Well-balanced classes → Standard multi-class approach
2. **Regression**: Full intensity range → Standard regression appropriate
3. **Algorithm Choice**: Near-zero linear correlations → **Gradient Boosting** (non-linear)
4. **Feature Engineering**:
   - Text features likely most predictive
   - Interaction features capture hidden patterns
   - One-hot encoding preserves category information

### Preprocessing Pipeline

```
Raw Journal Entry
       ↓
[Text Pipeline] → cleaned_text, sentiment, quality metrics
       ↓
[Numerical Engineering] → normalized values, interactions
       ↓
[Categorical Encoding] → one-hot vectors
       ↓
[Feature Fusion] → 48-dimensional feature vector
       ↓
[Model Input] → Ready for training/inference
```

---

## ▶️ Next Steps: Day 4 - Model Training

### Day 4 Tasks

- [ ] Task 4.1: Train Emotional State Classifier (6-class)
- [ ] Task 4.2: Train Intensity Regressor (1-5 scale)
- [ ] Task 4.3: Evaluate model performance
- [ ] Task 4.4: Save trained models for inference

### Success Criteria for Day 4

- [ ] Classifier accuracy > 20% (baseline: 16.7% random)
- [ ] Regressor MAE < 2.0 (baseline: ~1.4 random)
- [ ] Models train in < 5 minutes on local machine
- [ ] Model files < 50MB each (edge deployment target)

### Algorithm: Gradient Boosting

**Why?**

- ✅ Handles mixed feature types natively
- ✅ Captures non-linear relationships
- ✅ Robust to feature scaling variations
- ✅ Fast inference (critical for edge deployment)
- ✅ Built-in feature importance (for interpretability)

---

## 🛠️ How to Run Scripts

### Prerequisites

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify dependencies
pip list
```

### Run Full Pipeline (Days 1-3)

```powershell
# Data validation
python src/data_validation.py

# EDA (run individually)
python src/eda_step1.py
python src/eda_step2.py
python src/eda_step3.py
python src/eda_step4.py
python src/eda_step5.py

# Text pipeline
python src/text_pipeline.py

# Feature engineering
python src/numerical_features.py
python src/categorical_features.py
python src/feature_fusion.py
```

### View Outputs

```powershell
# Markdown reports
code outputs/data_validation_report.md

# PNG visualizations
Start-Process outputs/emotional_state_dist.png

# Check feature matrix
python -c "import pandas as pd; df = pd.read_csv('data/train_data.csv'); print(df.shape)"
```

---

## 📦 Dependencies

Core packages (see `requirements.txt`):

```txt
pandas==2.0.3          # Data manipulation
numpy==1.24.3          # Numerical operations
scikit-learn==1.3.0    # Machine learning models
matplotlib==3.7.2      # Basic plotting
seaborn==0.12.2        # Statistical visualizations
vaderSentiment==3.3.2  # Text sentiment analysis
```

Install with:

```powershell
pip install -r requirements.txt
```

---

## 🤝 Contributing

1. Follow the one-task-at-a-time approach
2. Document all findings in this README or separate reports
3. Test scripts before marking tasks complete
4. Share outputs for review before proceeding

---

## 📝 Notes

- All processing designed for **local/offline execution**
- Model size target: **<100MB total** for edge deployment
- Privacy: **No data leaves the device** in final implementation
- Uncertainty quantification is **central to decision logic**
- Feature matrix: **48 features** ready for model training

---

_Last updated: Day 3 Complete - Feature Engineering & Fusion ✅_

_Next: Day 4 - Model Training_

````

---

## **✅ How to Use This**

1. **Open** `README.md` in your code editor
2. **Delete all existing content**
3. **Paste** the complete content above
4. **Save** the file (`Ctrl+S`)
5. **Verify** it exists:
   ```powershell
   Test-Path "README.md"
````

---
