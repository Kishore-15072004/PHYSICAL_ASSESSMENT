# 📊 Physical Education Assessment System - Complete Code Analysis

**Project Overview:** An AI-powered Physical Education (PE) assessment system that predicts student PE scores (0-100) using machine learning and deep learning with 17 multidimensional attributes (physical metrics + psychological/social indicators).

---

## 🏗️ Project Structure

```
Physical_Assessment/
├── base_model/                    # Original implementation (7 features)
│   ├── bpnn_predictor.py
│   ├── step2_preprocessing_spark.py
│   ├── step3_feature_analysis_spark.py
│   ├── step4_bpnn_model.py
│   ├── test.py
│   └── data/
│       ├── pe_assessment_dataset.csv
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── Upgraded_model/                # Enhanced implementation (17 features + ensemble)
│   ├── bpnn_predictor.py
│   ├── step2_preprocessing_spark.py
│   ├── step3_feature_analysis_spark.py
│   ├── step4_bpnn_model.py
│   ├── step5_ensemble_ml_model.py     ⭐ New: 6-model ensemble
│   ├── demo.py
│   ├── model_evalutation.py
│   ├── prediction.py
│   ├── recommendation_engine.py
│   ├── requirements.txt
│   ├── README.md
│   └── data/ + visualizations/
│
└── data/                          # Shared datasets
    ├── train_processed.csv
    └── test_processed.csv
```

---

## 🔍 Detailed Code Analysis

### **1️⃣ BASE MODEL - Foundation Implementation**

#### `base_model/step2_preprocessing_spark.py`
**Purpose:** Data preprocessing and initial feature engineering using Apache Spark

**Key Functions:**
- Loads PE dataset from CSV into Spark DataFrame
- Creates **7 core physical features** through feature engineering
- Generates synthetic **psycho-social attributes** (motivation, stress_level, self_confidence, focus, teamwork, peer_support, communication)
- Uses `least()` and `greatest()` functions to bound feature values
- Applies **Min-Max scaling** (0-1 normalization) to all features
- Outputs processed data: `train_processed.csv` and `test_processed.csv`

**Feature Engineering Logic:**
```python
# Example: motivation derived from participation
motivation = min(max(participation/10 + random, 4), 9)

# stress_level inversely related to progress
stress_level = min(max(8 - physical_progress/15 + random, 2), 8)
```

**Key Libraries:** PySpark SQL, pandas, scikit-learn VectorAssembler/MinMaxScaler

---

#### `base_model/step3_feature_analysis_spark.py`
**Purpose:** Statistical analysis and correlation studies

**Key Functions:**
- Computes summary statistics (mean, min, max, std) using Spark
- Generates **correlation coefficients** between each feature and target score
- Creates scatter plots for each feature vs PE Score
- Exports correlation data to CSV for later visualization
- Samples data (25%) to Pandas for visualization efficiency

**Output Files:**
- `summary_statistics.csv` - Statistical summaries
- `feature_correlation_values.csv` - Correlation coefficients
- Feature vs Score scatter plots

---

#### `base_model/step4_bpnn_model.py`
**Purpose:** Train Back-Propagation Neural Network (BPNN) from scratch

**Neural Network Architecture:**
```
Input Layer (7 features) 
    ↓
Hidden Layer (12 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, Linear activation)
```

**Training Details:**
- **Algorithm:** Mini-batch gradient descent with backpropagation
- **Learning Rate:** 0.0005
- **Epochs:** 200
- **Batch Size:** 256
- **Activation:** ReLU in hidden layer
- **Loss Function:** Mean Squared Error (MSE)
- **Optimization:** Gradient clipping (clip_value=1.0) to prevent exploding gradients

**Backpropagation Equations:**
```
Forward Pass:
  z1 = X·W1 + b1
  a1 = ReLU(z1)
  z2 = a1·W2 + b2   (output)

Backward Pass:
  error = z2 - y_true
  dW2 = a1^T · error
  da1 = error · W2^T
  dz1 = da1 ⊙ ReLU_derivative(z1)
  dW1 = X^T · dz1
```

**Output:** Saves trained weights to `saved_model/bpnn_model.npz`

---

#### `base_model/bpnn_predictor.py`
**Purpose:** Inference module for making predictions

**Key Functions:**
- `load_bpnn_model()` - Loads saved model weights
- `min_max_scale()` - Scales new inputs to [0,1]
- `predict_pe_score()` - Makes prediction on 7 physical features

**Prediction Pipeline:**
```python
features (raw) → min_max_scale → forward_pass → clip(0,100) → final_score
```

---

#### `base_model/test.py`
**Purpose:** Simple testing script

**Functionality:**
- Loads test dataset
- Runs inference on test set
- Prints sample predictions (first 10 predictions)

---

### **2️⃣ UPGRADED MODEL - Enhanced System (Recommended)**

#### Key Improvements Over Base Model

| Aspect | Base Model | Upgraded Model |
|--------|-----------|-----------------|
| **Features** | 7 physical metrics | 17 features (7 physical + 10 psychological) |
| **Models** | 1 BPNN only | 6-model ensemble |
| **Hidden Neurons** | 12 | 16 (deeper network) |
| **Ensemble Methods** | N/A | BPNN + RF + GB + SVR + LR + XGBoost |
| **Accuracy (R² Score)** | 0.812 | 0.811 (Ensemble), 0.815 (Linear Regression best) |
| **RMSE** | 3.01 | 3.03 (Ensemble weighted average) |
| **Explainability** | None | SHAP analysis |
| **Recommendations** | None | AI-powered coaching via Groq LLM |

**17 Features in Upgraded Model:**
- **Compulsory (7):** attendance, endurance, strength, flexibility, participation, skill_speed, physical_progress
- **Optional (10):** motivation, stress_level, self_confidence, focus, teamwork, peer_support, communication, sleep_quality, nutrition, screen_time

---

#### `Upgraded_model/step2_preprocessing_spark.py`
**Same as base_model** but generates 17 features instead of 7

**New Features Added:**
```python
motivation, stress_level, self_confidence, focus, teamwork, 
peer_support, communication, sleep_quality, nutrition, screen_time
```

Each generated through correlations with base features and random noise to simulate real-world variation.

---

#### `Upgraded_model/step3_feature_analysis_spark.py`
**Same as base_model** but analyzes all 17 features

**Output:** Comprehensive correlation heatmap showing relationships between all features

---

#### `Upgraded_model/step4_bpnn_model.py`
**Enhanced BPNN Training**

**Architecture Differences:**
```
Input Layer (17 features) ← Upgraded from 7
    ↓
Hidden Layer (16 neurons, ReLU) ← Increased from 12
    ↓
Output Layer (1 neuron)
```

**Same hyperparameters as base**, but trained on 17 features

**Performance Metrics:**
- R² Score (Train): ~0.81
- R² Score (Test): ~0.81
- RMSE: ~3.01

---

#### ⭐ `Upgraded_model/step5_ensemble_ml_model.py` - THE KEY UPGRADE

**Purpose:** Train 6 complementary ML models and combine via weighted ensemble

**Models Implemented:**

1. **BPNN** (Deep Learning)
   - Captures non-linear relationships
   - R² Score: 0.812

2. **Random Forest** (Ensemble Tree)
   - 100 trees, max_depth=15
   - Robust to outliers
   - R² Score: 0.679

3. **Gradient Boosting** (Sequential Tree)
   - n_estimators=100, learning_rate=0.1
   - Builds on errors from previous trees
   - R² Score: 0.740

4. **Support Vector Regression (SVR)** (Kernel-based)
   - RBF kernel, C=100
   - Works in high-dimensional space
   - R² Score: 0.782

5. **Linear Regression** (Baseline)
   - Simple linear model
   - Fast inference, interpretable
   - R² Score: 0.815 ⭐ **BEST SINGLE MODEL**

6. **XGBoost** (Advanced Gradient Boosting)
   - n_estimators=100, max_depth=5, learning_rate=0.1
   - State-of-the-art gradient boosting
   - R² Score: 0.742

**Ensemble Strategy:**
```
Weighted Ensemble = w1·BPNN + w2·RF + w3·GB + w4·SVR + w5·LR + w6·XGBoost

Default Weights: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
                 (Equal weight to all 6 models)
```

**Why Ensemble?**
- ✅ Combines strengths of diverse model families
- ✅ Reduces variance through averaging
- ✅ Better generalization than single model
- ✅ Robust to outliers and noisy data
- ✅ Ensemble R² Score: 0.788

**Training Process:**
```python
1. Load preprocessed 17-feature data
2. Train BPNN (already exists from step4)
3. Train 5 sklearn models on same data
4. Generate predictions from all 6 models
5. Calculate model correlations
6. Optimize and save ensemble weights
7. Create visualizations:
   - Model performance comparison (R² scores)
   - Prediction correlation matrix
   - RMSE comparison
```

**Saved Artifacts:**
```
saved_model/ensemble/
├── random_forest.pkl
├── gradient_boosting.pkl
├── svr.pkl
├── linear_regression.pkl
├── xgboost.pkl
└── ensemble_info.pkl      (weights & metadata)
```

---

#### `Upgraded_model/bpnn_predictor.py`
**Enhanced Inference Module**

**Key Functions:**
- `load_ensemble_models()` - Loads all 6 models + weights
- Predicts using both BPNN and sklearn ensemble
- Similar structure to base model but loads multiple models

**Feature Configuration:**
```python
# Compulsory features (must provide)
compulsory_features = [
    ("attendance", 50, 100),
    ("endurance", 30, 95),
    ("strength", 35, 95),
    ("flexibility", 30, 90),
    ("participation", 55, 100),
    ("skill_speed", 30, 95),
    ("physical_progress", 40, 95),
]

# Optional features (defaults to 6 if missing)
optional_features = [
    ("motivation", 1, 10),
    ("stress_level", 1, 10),
    ...
]
```

---

#### ⭐ `Upgraded_model/prediction.py` - Interactive Prediction Interface

**Purpose:** User-friendly prediction system with explainability

**Workflow:**
1. Collect user inputs (7 compulsory + 10 optional features)
2. Use min-max scaling based on feature ranges
3. Make predictions from all 6 models
4. Compute weighted ensemble prediction
5. Generate SHAP explanations (feature importance)
6. Create personalized recommendations
7. Generate diagnostic report

**Key Features:**
- **Descriptive Input Options:** For complex metrics (stress_level, sleep_quality)
  ```python
  stress_level options:
    - "Low (Calm/Relaxed)" → 2
    - "Moderate (Manageable)" → 5
    - "High (Overwhelmed)" → 8
  ```

- **SHAP Explainability:** 
  - Shows which features most influenced the prediction
  - Positive/negative contributions
  - Helps understand model decisions

- **Feature Ranges Validation:**
  - Compulsory features must be within specified ranges
  - Optional features can use defaults if not provided
  - Input validation prevents invalid values

---

#### ⭐ `Upgraded_model/recommendation_engine.py` - AI Coach

**Purpose:** Generate personalized coaching recommendations

**Architecture:**
```python
class RecommendationEngine:
    1. Analyze user profile (identify strengths/weaknesses)
    2. Generate AI recommendations via Groq LLM (optional)
    3. Fallback to static, evidence-based recommendations
    4. Combine top AI + static tips
    5. Return in structured format
```

**Key Features:**

1. **Profile Analysis:**
   - Identifies strengths (above threshold) and weaknesses
   - Compares against feature-specific thresholds
   - Example: attendance < 80 is weak

2. **Thresholds by Feature:**
   ```python
   thresholds = {
       "attendance": 80,
       "endurance": 60,
       "strength": 60,
       ...
       "stress_level": 6,  (inverted: higher is worse)
   }
   ```

3. **Two-Source Strategy:**
   
   **A) AI Tips (via Groq LLM):**
   - Queries LLM with: PE Score + Weaknesses
   - Prompt: "Give 2 FITT coaching tips"
   - FITT = Frequency, Intensity, Time, Type
   - Example: "Run 3x weekly for 20 min at moderate intensity"
   
   **B) Static Tips (Evidence-based):**
   - Hardcoded recommendations for 6 features
   - Example:
     ```python
     "endurance": "Perform 20 minutes of light jogging 3x per week."
     "stress_level": "Practice Box Breathing: 4s inhale, 4s hold, 4s exhale"
     ```

4. **Fallback Logic:**
   - If Groq API unavailable → use static tips only
   - If no weak areas → focus on optimization

**Output Format:**
```python
{
    "strengths": ["Attendance", "Participation"],
    "weak_areas": ["Endurance", "Flexibility"],
    "recommendations": [
        "AI tip 1",
        "AI tip 2",
        "Static tip 1",
        "Static tip 2"
    ]
}
```

---

#### ⭐ `Upgraded_model/model_evalutation.py` - Comprehensive Model Evaluation

**Purpose:** Test all 6 models and generate performance metrics

**Metrics Calculated:**
- **R² Score** (coefficient of determination)
  - Measures how well model explains variance
  - 1.0 = perfect, 0.0 = no better than mean
  
- **RMSE** (Root Mean Squared Error)
  - Penalizes larger errors more heavily
  - Scale: 0-100 (PE score range)
  - Lower is better
  
- **MAE** (Mean Absolute Error)
  - Average absolute prediction error
  - More interpretable than RMSE

**Output Visualizations:**
- Performance comparison bar chart (R² scores)
- RMSE comparison by model
- Prediction correlation heatmap (how similar are models to each other?)
- Residual plots (prediction errors)
- Actual vs Predicted scatter plot

**Key Findings:**
```
Best Individual Model: Linear Regression (R² = 0.815)
Best Ensemble: Top 3 average (R² = 0.811)
Ensemble Advantage: Robustness over raw accuracy
```

---

#### ⭐ `Upgraded_model/demo.py` - Groq LLM Integration Test

**Purpose:** Verify Groq API connectivity (optional)

**Three Models Tested:**
1. `llama-3.3-70b-versatile` (primary)
2. `meta-llama/llama-4-scout-17b-16e-instruct`
3. `llama-3.1-8b-instant` (fallback)

**Usage:**
```bash
# Requires: pip install groq
# Requires: GROQ_API_KEY in .env file
python demo.py
```

**Output:** Confirms which model is available for recommendation engine

---

#### `Upgraded_model/requirements.txt`

**Dependencies:**
```
numpy>=1.23,<2.0           # Numerical computing
pandas>=1.5,<2.0           # Data manipulation
matplotlib>=3.5,<4.0       # Visualization
scikit-learn>=1.2,<2.0     # Machine learning (RF, GB, SVR, LR)
xgboost>=1.7,<2.0          # Gradient boosting
shap>=0.41,<0.50           # Model explainability
pyspark>=3.3,<4.0          # Distributed processing
findspark>=2.0             # Spark finder
python-dotenv>=1.0         # Environment variables
```

---

#### `Upgraded_model/README.md`

**Comprehensive Documentation:**
- Project overview & motivation
- System architecture (big-data ready)
- Model justifications & theoretical foundation
- Installation & usage instructions
- API documentation
- Example predictions & interpretations

---

## 🔄 Data Pipeline Flow

```
Raw PE Assessment Dataset (pe_assessment_dataset.csv)
        ↓
   [Step 2: Preprocessing]
   - Load via Spark
   - Generate 17 features
   - Min-Max scaling
   - Train/Test split (80/20)
        ↓
train_processed.csv + test_processed.csv
        ↓
   [Step 3: Feature Analysis]
   - Correlation analysis
   - Statistical summaries
   - Scatter plot visualizations
        ↓
correlation_values.csv + visualization files
        ↓
   [Step 4: Model Training]
   - BPNN training (17 features)
   - Backpropagation optimization
        ↓
saved_model/bpnn_model.npz
        ↓
   [Step 5: Ensemble Creation]
   - Train 5 sklearn models
   - Weight combination
   - Evaluate ensemble
        ↓
saved_model/ensemble/*.pkl + ensemble_info.pkl
        ↓
   [Inference]
   - Load all 6 models
   - prediction.py for user interaction
   - SHAP explanations
   - Recommendations via recommendation_engine.py
        ↓
Final PE Score + Explanation + Coaching Tips
```

---

## 📊 Model Performance Summary

### Individual Model Performance:
| Model | R² Score | RMSE | MAE |
|-------|---------|------|-----|
| Linear Regression | **0.815** | 2.99 | 2.24 |
| BPNN | 0.812 | 3.01 | 2.25 |
| Ensemble (Top 3) | 0.811 | 3.03 | 2.26 |
| SVR | 0.782 | 3.25 | 2.43 |
| XGBoost | 0.742 | 3.54 | 2.65 |
| Gradient Boosting | 0.740 | 3.55 | 2.66 |
| Random Forest | 0.679 | 3.94 | 2.95 |

### Key Insights:
- ✅ Linear model surprisingly effective (well-scaled features)
- ✅ BPNN close to linear regression (good generalization)
- ✅ Ensemble provides stability through diversity
- ✅ Tree models slightly underperform (may need tuning)
- ✅ All models show good vs poor predictions correlation (0.95+)

---

## 🎯 Feature Importance (from correlation analysis):

**Strongest Correlations with PE Score:**
1. **attendance** - 0.95+ correlation (strongest)
2. **endurance** - 0.92+ correlation
3. **strength** - 0.90+ correlation
4. **skill_speed** - 0.88+ correlation
5. **physical_progress** - 0.85+ correlation

**Moderate Correlations:**
- participation, flexibility, teamwork, self_confidence, focus

**Weak Correlations:**
- screen_time (negative), stress_level (negative, when high)

---

## 🚀 How to Run the System

### Option 1: Full Training Pipeline
```bash
# Install dependencies
pip install -r Upgraded_model/requirements.txt

# Run preprocessing
python Upgraded_model/step2_preprocessing_spark.py

# Run feature analysis
python Upgraded_model/step3_feature_analysis_spark.py

# Train BPNN
python Upgraded_model/step4_bpnn_model.py

# Train ensemble
python Upgraded_model/step5_ensemble_ml_model.py
```

### Option 2: Inference Only (Pre-trained Models)
```bash
# Interactive prediction with explanations
python Upgraded_model/prediction.py

# Evaluate performance
python Upgraded_model/model_evalutation.py
```

### Option 3: Programmatic Usage
```python
from bpnn_predictor import load_ensemble_models
from prediction import collect_user_inputs

# Load models once
models = load_ensemble_models()

# Make predictions
user_inputs = collect_user_inputs()  # Interactive
# Or manually: user_inputs = {feature: value, ...}

# Generate predictions
```

---

## 🔍 Code Quality & Design Patterns

### ✅ Strengths:
1. **Modular Design:** Each step is independent
2. **Big-Data Ready:** Uses Spark for scalability
3. **Reproducibility:** Fixed random seeds (np.random.seed(42))
4. **Documentation:** README with equations and architecture
5. **Ensemble Diversity:** 6 fundamentally different model types
6. **Explainability:** SHAP integration for interpretability
7. **Error Handling:** Try-except blocks for model loading
8. **Predictions:** Multiple output formats

### ⚠️ Areas for Improvement:
1. **Type Hints:** No type annotations in Python files
2. **Logging:** Minimal structured logging; mostly prints
3. **Testing:** No unit tests visible
4. **Config File:** Hardcoded hyperparameters (no YAML config)
5. **Validation:** Limited input validation
6. **Error Messages:** Could be more descriptive
7. **Caching:** Could cache model weights to memory
8. **Async:** No async/parallel prediction capabilities

---

## 📈 Key Takeaways

1. **Comprehensive System:** From preprocessing → training → inference → explainability
2. **Scalable Architecture:** Built on Spark for big-data scenarios
3. **Production-Ready:** Error handling, model persistence, documentation
4. **Interpretable:** Combines accuracy with SHAP explanations
5. **User-Centric:** Interactive prediction + personalized recommendations
6. **Ensemble Approach:** Balances accuracy with robustness
7. **Well-Documented:** Clear README and code comments

---

## 🔗 File Dependencies

```
prediction.py
    ↓ imports
├── bpnn_predictor.py (load_ensemble_models)
├── recommendation_engine.py (generate_recommendations)
└── shap (for explanations)

step5_ensemble_ml_model.py
    ↓ requires
├── step4_bpnn_model.py output (bpnn_model.npz)
└── Preprocessed data (train_processed.csv, test_processed.csv)

model_evalutation.py
    ↓ requires
├── All saved models in saved_model/ensemble/
└── bpnn_model.npz

demo.py (standalone, optional)
    ↓ requires
└── GROQ_API_KEY in .env
```

---

**Generated:** March 10, 2026
**System Version:** 2.0 (Upgraded Model with Ensemble)
