# Emotional Intelligence AI System

> Predict emotional states and intensity from journal reflections, then provide personalized recommendations.

## 📋 Project Status

**Current Phase**: Day 1 - Exploratory Data Analysis (EDA) ✅ Complete

**Last Updated**: $(date)

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
├── 📄 README.md                 # This file
├── 📄 requirements.txt          # Python dependencies
│
├── 📁 data/
│   ├── 📄 train_data.csv       # Training data (1200 rows, labeled)
│   └── 📄 test_data.csv        # Test data (120 rows, unlabeled)
│
├── 📁 src/
│   ├── 📄 data_validation.py   # Day 1: Validate dataset structure
│   ├── 📄 eda_step1.py         # Day 1: Emotional state distribution
│   ├── 📄 eda_step2.py         # Day 1: Intensity distribution
│   ├── 📄 eda_step3.py         # Day 1: Numerical features histograms
│   ├── 📄 eda_step4.py         # Day 1: Correlation heatmap
│   └── 📄 eda_step5.py         # Day 1: Text length analysis
│
├── 📁 outputs/
│   ├── 📄 data_validation_report.md  # Validation results
│   ├── 📄 emotional_state_dist.png   # EDA visualization
│   ├── 📄 intensity_dist.png         # EDA visualization
│   ├── 📄 numerical_features_dist.png # EDA visualization
│   ├── 📄 correlation_heatmap.png    # EDA visualization
│   └── 📄 text_length_analysis.png   # EDA visualization
│
├── 📁 models/                  # (Future) Trained model files
├── 📁 tests/                   # (Future) Test scripts
└── 📁 analysis/                # (Future) Error analysis reports
```

---

## 🚀 Completed Tasks (Day 1)

### ✅ Task 1.1-1.3: Environment Setup

- [x] Project folder structure created
- [x] Virtual environment configured
- [x] Dependencies installed via `requirements.txt`

### ✅ Task 1.4: Data Validation

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
Data types: Mixed (str, int64, float64)
```

**Output**: `outputs/data_validation_report.md`

### ✅ Task 1.5: Exploratory Data Analysis

#### Step 1: Emotional State Distribution

**Script**: `src/eda_step1.py`

**Findings**:

```
calm:        216 samples (18.0%)
restless:    209 samples (17.4%)
neutral:     201 samples (16.8%)
focused:     193 samples (16.1%)
mixed:       191 samples (15.9%)
overwhelmed: 190 samples (15.8%)
```

✅ Classes are well-balanced (imbalance ratio: 1.14)

**Output**: `outputs/emotional_state_dist.png`

#### Step 2: Intensity Distribution

**Script**: `src/eda_step2.py`

**Findings**:

```
Mean: 3.05, Median: 3.0, Std Dev: 1.39
Range: [1, 5] with good coverage across all levels
```

✅ Suitable for regression approach

**Output**: `outputs/intensity_dist.png`

#### Step 3: Numerical Features Histograms

**Script**: `src/eda_step3.py`

**Findings**:

```
duration_min:  Mean=15.9, Range=[3, 35], No missing
sleep_hours:   Mean=6.0, Range=[3.5, 8.5], 7 missing
energy_level:  Mean=3.0, Range=[1, 5], Scale=1-5 ⚠️
stress_level:  Mean=3.0, Range=[1, 5], Scale=1-5 ⚠️
```

⚠️ **Important**: Energy and stress use 1-5 scale (not 1-10)

**Output**: `outputs/numerical_features_dist.png`

#### Step 4: Correlation Heatmap

**Script**: `src/eda_step4.py`

**Findings**:

```
Correlations with intensity:
  duration_min:  -0.016 (weak negative)
  sleep_hours:   -0.034 (weak negative)
  energy_level:  -0.005 (weak negative)
  stress_level:   0.003 (weak positive)
```

🎯 **Key Insight**: Numerical features have almost NO linear correlation with intensity
→ Need non-linear models (Gradient Boosting) to find patterns

**Output**: `outputs/correlation_heatmap.png`

#### Step 5: Text Length Analysis

**Script**: `src/eda_step5.py`

**Findings**:

```
Characters: Mean=58.3, Median=61, Range=[9, 169]
Words:      Mean=10.9, Median=11, Range=[2, 32]
Short texts (≤5 words): ~4% of dataset
```

✅ Most texts are moderate length; short texts exist but are minority

**Output**: `outputs/text_length_analysis.png`

---

## 🔑 Key Insights from EDA

### Data Quality

| Aspect          | Status        | Notes                                  |
| --------------- | ------------- | -------------------------------------- |
| Class Balance   | ✅ Excellent  | All 6 emotional states ~16-18%         |
| Intensity Range | ✅ Good       | Full 1-5 scale covered                 |
| Missing Values  | ⚠️ Manageable | Only 3 columns affected, max 10%       |
| Feature Scales  | ⚠️ Note       | Energy/stress use 1-5 scale (not 1-10) |

### Modeling Implications

1. **Classification**: Well-balanced classes → No need for class weights
2. **Regression**: Full intensity range → Standard regression appropriate
3. **Feature Engineering**:
   - Low linear correlations → Need non-linear models
   - Text features likely more predictive than numerical
   - Feature interactions may be important
4. **Preprocessing**:
   - Handle 10% missing `face_emotion_hint` with imputation
   - Account for short texts (~4%) in uncertainty modeling
   - Normalize energy/stress to 1-5 scale consistently

---

## ▶️ Next Steps (Day 2 Preview)

### Day 2: Text Preprocessing Module

- [ ] Create text cleaner (handle lowercase, punctuation, emojis)
- [ ] Implement sentiment analysis (VADER)
- [ ] Add text quality metrics (length, coherence, ambiguity)
- [ ] Test with edge cases (very short text, contradictions)

### Day 3: Feature Engineering

- [ ] Engineer text features (9 features)
- [ ] Engineer numerical features (8 features with interactions)
- [ ] Encode categorical features (22 features after one-hot)
- [ ] Create feature fusion pipeline

### Day 4: Model Training

- [ ] Train emotional state classifier (GradientBoosting)
- [ ] Train intensity regressor (GradientBoosting)
- [ ] Evaluate baseline performance
- [ ] Save models for inference

---

## 🛠️ How to Run Scripts

### Prerequisites

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify dependencies
pip list
```

### Run Data Validation

```bash
python src/data_validation.py
# Output: Console + outputs/data_validation_report.md
```

### Run EDA Scripts (One at a Time)

```bash
python src/eda_step1.py  # Emotional state distribution
python src/eda_step2.py  # Intensity distribution
python src/eda_step3.py  # Numerical features histograms
python src/eda_step4.py  # Correlation heatmap
python src/eda_step5.py  # Text length analysis
```

### View Output Files

```bash
# Open markdown report in any text editor
code outputs/data_validation_report.md

# View PNG images
Start-Process outputs/emotional_state_dist.png
```

---

## 📦 Dependencies

See `requirements.txt` for full list. Core packages:

```txt
pandas==2.0.3          # Data manipulation
numpy==1.24.3          # Numerical operations
scikit-learn==1.3.0    # Machine learning models
matplotlib==3.7.2      # Basic plotting
seaborn==0.12.2        # Statistical visualizations
vaderSentiment==3.3.2  # Text sentiment analysis
```

Install with:

```bash
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

- All processing is designed for **local/offline execution**
- Model size target: **<100MB** for edge deployment
- Privacy: **No data leaves the device** in final implementation
- Uncertainty quantification is **central to decision logic**
