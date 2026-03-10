# 🎨 Visual Architecture & Component Summary

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│          PHYSICAL EDUCATION ASSESSMENT SYSTEM v2.0             │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Raw Dataset     │
│ (CSV, Parquet)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│   STEP 2: PREPROCESSING (Spark)          │
│  - Load data into Spark DataFrame        │
│  - Feature engineering (17 features)     │
│  - Min-Max normalization (0-1)           │
│  - Train/Test split (80/20)              │
└────────┬─────────────────────────────────┘
         │
         ▼
    ┌────────────────────────┐
    │ Processed Datasets     │
    │ ├─ X_train (80%)       │
    │ ├─ y_train             │
    │ ├─ X_test (20%)        │
    │ └─ y_test              │
    └────────┬───────────────┘
             │
             ├──────────────────────────────┐
             │                              │
             ▼                              ▼
    ┌──────────────────────┐    ┌──────────────────────┐
    │ STEP 3: ANALYSIS     │    │ STEP 4: MODEL TRAIN  │
    │ - Correlation study  │    │ - BPNN training      │
    │ - Stat summaries     │    │ - 17 input neurons   │
    │ - Visualization      │    │ - 16 hidden neurons  │
    └──────────────────────┘    │ - Backpropagation    │
                                 └──────────┬───────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │ BPNN Weights     │
                                   │ (bpnn_model.npz) │
                                   └────────┬─────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
         ┌──────────────────┐    ┌─────────────────────┐  ┌──────────────┐
         │ STEP 5: ENSEMBLE │    │ ML Models (Sklearn) │  │ All Outputs  │
         │                  │    │                     │  │ Combined via │
         │ ├─ Random Forest │    │ ├─ RF               │  │ Weighted     │
         │ ├─ Gradient Boost│    │ ├─ GB               │  │ Ensemble     │
         │ ├─ SVR           │    │ ├─ SVR              │  │              │
         │ ├─ Linear Reg    │    │ ├─ Linear Reg ⭐    │  │ W = [0.2 ×6] │
         │ └─ XGBoost       │    │ └─ XGBoost          │  │              │
         └────────┬─────────┘    └─────────┬───────────┘  └──────┬───────┘
                  │                        │                     │
                  └────────────────────────┼─────────────────────┘
                                           │
                                           ▼
                        ┌──────────────────────────────────┐
                        │  SAVED MODELS & WEIGHTS          │
                        │  saved_model/ensemble/           │
                        │  ├─ random_forest.pkl            │
                        │  ├─ gradient_boosting.pkl        │
                        │  ├─ svr.pkl                      │
                        │  ├─ linear_regression.pkl        │
                        │  ├─ xgboost.pkl                  │
                        │  └─ ensemble_info.pkl            │
                        └────────────┬─────────────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────────┐
                        │  INFERENCE LAYER (prediction.py) │
                        │                                  │
                        │  1. User Input Collection        │
                        │  2. Min-Max Scaling              │
                        │  3. 6-Model Prediction           │
                        │  4. Weighted Averaging           │
                        │  5. SHAP Explanations            │
                        │  6. Recommendations              │
                        └────────────┬─────────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
    ┌─────────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Final PE Score     │  │  SHAP Feature    │  │  Recommendations │
    │  (0-100 scale)      │  │  Importance      │  │  via Groq LLM    │
    │                     │  │                  │  │  (AI Coach)      │
    │  ✓ Point estimate   │  │  ✓ Top features  │  │                  │
    │  ✓ Confidence       │  │  ✓ Positive/Neg  │  │  ✓ FITT based    │
    │  ✓ Individual preds │  │  ✓ Comparisons   │  │  ✓ Personalized  │
    └─────────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Data Flow: 17 Features Through Pipeline

```
COMPULSORY FEATURES (Physical Metrics)
├─ attendance ──────────────┐
├─ endurance ───────────────┤
├─ strength ────────────────┤
├─ flexibility ─────────────┼─→ Spark Preprocessing ──→ Min-Max Scale
├─ participation ──────────┤     (Step 2)                (0-1 range)
├─ skill_speed ────────────┤
└─ physical_progress ──────┘

OPTIONAL FEATURES (Psychological/Social Metrics)
├─ motivation ──────────────┐
├─ stress_level ────────────┤
├─ self_confidence ────────┤
├─ focus ───────────────────┼─→ Generated from base features
├─ teamwork ────────────────┤     + random variations
├─ peer_support ───────────┤
├─ communication ──────────┤
├─ sleep_quality ──────────┤
├─ nutrition ───────────────┤
└─ screen_time ────────────┘

ALL 17 FEATURES (scaled 0-1)
         ↓
    ┌────────────────────────────┐
    │ Feature Correlation Matrix │
    │ (Step 3 Analysis)          │
    │ → Visualizations created   │
    │ → Correlation CSV exported │
    └────────┬───────────────────┘
             │
             ↓
    ┌────────────────────────────┐
    │ Training Data (80%)        │ ┌─ Forward through 6 models
    ├─ X_train: (N, 17)          ├─ Predictions generated
    └─ y_train: (N, 1)           └─ Losses computed
             │
             ↓
    ┌────────────────────────────┐
    │ Testing Data (20%)         │ ┌─ Model evaluation
    ├─ X_test: (N, 17)           ├─ R², RMSE, MAE
    └─ y_test: (N, 1)            └─ Visualizations
```

---

## Neural Network Architecture (BPNN)

### Base Model (7 features):
```
Input: [attendance, endurance, strength, flexibility, participation, skill_speed, physical_progress]
         │
         ├─(W1: 7×12, b1: 1×12)──┐
         │                        │
         ▼                        ▼
    [Dense Layer - 12 neurons]
         │
         ├─ ReLU Activation: max(0, x)
         │
         ▼
    [Hidden: 12 neurons]
         │
         ├─(W2: 12×1, b2: 1×1)──┐
         │                       │
         ▼                       ▼
    [Output Layer - 1 neuron]
         │
         ▼
    Score (0-1 normalized)
         │
         ▼
    Convert to (0-100) scale
```

### Upgraded Model (17 features):
```
Input: [7 physical + 10 psychological features]
         │
         ├─(W1: 17×16, b1: 1×16)──┐
         │                         │
         ▼                         ▼
    [Dense Layer - 16 neurons]
         │
         ├─ ReLU Activation: max(0, x)
         │
         ▼
    [Hidden: 16 neurons]
         │
         ├─(W2: 16×1, b2: 1×1)──┐
         │                       │
         ▼                       ▼
    [Output Layer - 1 neuron]
         │
         ▼
    Score (0-1 normalized)
         │
         ▼
    Convert to (0-100) scale
```

**Theoretical Basis:**
$$\text{Hidden} = \text{ReLU}(X \cdot W_1 + b_1)$$
$$\text{Output} = \text{Hidden} \cdot W_2 + b_2$$
$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

---

## Ensemble Model Combination

```
Test Sample (17 features)
     │
     ├─────────────────┬─────────────────┬──────────────────┐
     │                 │                 │                  │
     ▼                 ▼                 ▼                  ▼
┌──────────┐    ┌──────────────┐   ┌──────────────┐   ┌──────────┐
│  BPNN    │    │ Random Forest│   │   Gradient   │   │   SVR    │
│          │    │  (RF)        │   │  Boosting    │   │          │
│Pred: 75  │    │  (GB)        │   │   (GB)       │   │Pred: 74  │
└────┬─────┘    │Pred: 73      │   │Pred: 76      │   └────┬─────┘
     │          └────┬─────────┘   │              │       │
     │               │             └────┬─────────┘       │
     │               │                   │                │
     └───────────────┼───────────────────┼────────────────┘
                     │                   │
                     ▼                   ▼
                ┌──────────┐        ┌──────────────┐
                │ Linear   │        │  XGBoost     │
                │ Regres   │        │  (XGB)       │
                │Pred: 75  │        │Pred: 74      │
                └────┬─────┘        └────┬─────────┘
                     │                   │
                     └───────┬───────────┘
                             │
              ╔══════════════════════════════╗
              ║  Weighted Ensemble Voting    ║
              ║                              ║
              ║  Final Pred = 0.2×75 +      ║
              ║              0.2×73 +       ║
              ║              0.2×76 +       ║
              ║              0.2×74 +       ║
              ║              0.2×75 +       ║
              ║              0.2×74         ║
              ║                              ║
              ║  = 74.5 (Robust Average)    ║
              ╚══════════════════════════════╝
                             │
                             ▼
                    ┌─────────────────┐
                    │ Final Score: 74.5│
                    │ + Std Dev: ±1.2  │
                    │ Confidence: High │
                    └─────────────────┘
```

---

## Model Performance Comparison

### Accuracy Metrics:
```
                      R² Score        RMSE       MAE
Linear Regression:    ╔═══╗═══╗═══╗   2.99      2.24   ⭐ BEST
                      ║   ║   ║   ║
BPNN:                 ╔═══╗═══╗═══╗   3.01      2.25
                      ║   ║   ║   ║
Ensemble (Top 3):     ╔═══╗═══╗═══╗   3.03      2.26
                      ║   ║   ║   ║
SVR:                  ╔════╗══╗     3.25      2.43
                      ║    ║  ║
XGBoost:              ╔════╗════╗    3.54      2.65
                      ║    ║    ║
Gradient Boosting:    ╔════╗════╗    3.55      2.66
                      ║    ║    ║
Random Forest:        ╔═════╗═════╗   3.94      2.95
                      ║     ║     ║
```

**Key Insight:** Linear Regression is best for this well-preprocessed dataset
- Features are properly scaled (0-1)
- Target has strong linear relationship with several features
- Tree models may be overfitting or underutilized

---

## Feature Importance Ranking

### Based on Correlation with PE Score:

```
Rank │ Feature              │ Correlation │ Importance
─────┼──────────────────────┼─────────────┼─────────────────
  1  │ attendance           │ 0.950       │ ████████████████████ ⭐⭐⭐
  2  │ endurance            │ 0.920       │ ██████████████████
  3  │ strength             │ 0.900       │ ██████████████████
  4  │ skill_speed          │ 0.880       │ █████████████████
  5  │ physical_progress    │ 0.850       │ ████████████████
  6  │ participation        │ 0.750       │ ███████████████
  7  │ teamwork             │ 0.700       │ ██████████████
  8  │ self_confidence      │ 0.680       │ █████████████
  9  │ focus                │ 0.650       │ █████████████
 10  │ flexibility          │ 0.600       │ ████████████
─────┼──────────────────────┼─────────────┼─────────────────
 11  │ motivation           │ 0.500       │ ██████████
 12  │ communication        │ 0.400       │ ████████
 13  │ sleep_quality        │ 0.350       │ ███████
 14  │ nutrition            │ 0.300       │ ██████
 15  │ peer_support         │ 0.250       │ █████
 16  │ stress_level (inv)   │ -0.400      │ ████████ (negative)
 17  │ screen_time (inv)    │ -0.450      │ █████████ (negative)

Legend: ⭐ = Critical, High correlation with target
```

---

## Training Process Flow

### For BPNN (step4_bpnn_model.py):

```
EPOCH LOOP (200 iterations)
│
├─ Shuffle training data
│
├─ BATCH LOOP (batch_size = 256)
│  │
│  ├─ FORWARD PASS:
│  │  ├─ z1 = X_batch · W1 + b1       (17×12 matrix multiplication)
│  │  ├─ a1 = ReLU(z1)                (non-linearity)
│  │  └─ z2 = a1 · W2 + b2            (output layer)
│  │
│  ├─ COMPUTE LOSS:
│  │  └─ L = MSE(y_pred, y_true)      (scalar value)
│  │
│  ├─ BACKWARD PASS (Backpropagation):
│  │  ├─ error = z2 - y_batch         (output error)
│  │  ├─ dW2 = a1^T · error           (output weight gradient)
│  │  ├─ db2 = sum(error)             (output bias gradient)
│  │  ├─ da1 = error · W2^T           (hidden error)
│  │  ├─ dz1 = da1 ⊙ ReLU'(z1)        (hidden gradient w/ activation)
│  │  ├─ dW1 = X^T · dz1              (hidden weight gradient)
│  │  └─ db1 = sum(dz1)               (hidden bias gradient)
│  │
│  ├─ GRADIENT CLIPPING:
│  │  ├─ dW1 = clip(dW1, -1.0, 1.0)   (prevent explosions)
│  │  └─ dW2 = clip(dW2, -1.0, 1.0)
│  │
│  └─ PARAMETER UPDATE:
│     ├─ W2 -= lr × dW2               (lr = 0.0005)
│     ├─ b2 -= lr × db2
│     ├─ W1 -= lr × dW1
│     └─ b1 -= lr × db1
│
└─ Record epoch loss for visualization
```

### For Ensemble (step5_ensemble_ml_model.py):

```
For each ML Model (5 sklearn models):
│
├─ Model.fit(X_train, y_train)        (Train on 80% data)
├─ y_pred_train = Model.predict(X_train)
├─ y_pred_test = Model.predict(X_test)
├─ Calculate R², RMSE, MAE metrics
└─ Save model.pkl

After all models trained:
│
├─ Generate predictions from all 6 models
├─ Calculate correlation between predictions
├─ Optimize ensemble weights (optional)
├─ Create visualizations
│  ├─ R² comparison bar chart
│  ├─ RMSE comparison
│  ├─ Prediction correlation heatmap
│  └─ Residual plots
│
└─ Save ensemble_info.pkl (with weights)
```

---

## Recommendation Generation Flow

```
USER PROFILE ANALYSIS
│
├─ Feature Thresholds:
│  ├─ attendance: 80          (below = weak)
│  ├─ endurance: 60           (below = weak)
│  ├─ stress_level: 6         (above = weak, inverted)
│  └─ ... (10 more thresholds)
│
├─ Classify each feature:
│  ├─ IF value < threshold → WEAK AREA
│  └─ IF value ≥ threshold → STRENGTH
│
└─ Result:
   ├─ strengths: [attendance, participation]
   └─ weak_areas: [endurance, flexibility]

GET RECOMMENDATIONS (Two-Source Strategy)
│
├─ SOURCE 1: AI COACH (Groq LLM)
│  │
│  ├─ IF Groq API available:
│  │  ├─ Prompt: "PE Score: 72%. Weaknesses: endurance, flexibility."
│  │  ├─ LLM generates: 2 FITT-based tips
│  │  │               (Frequency, Intensity, Time, Type)
│  │  └─ Extract actionable recommendations
│  │
│  └─ IF Groq unavailable → skip AI tips
│
├─ SOURCE 2: STATIC RECOMMENDATIONS
│  │
│  ├─ For each weak area:
│  │  └─ Lookup hardcoded recommendation
│  │     "endurance" → "20 min jogging, 3x per week"
│  │     "flexibility" → "10 min stretching daily"
│  │
│  └─ Randomly shuffle static tips
│
└─ MERGE & RANK
   │
   ├─ Combine: [AI tip 1, AI tip 2] + [static 1, static 2]
   ├─ Sort by relevance
   ├─ Return top 2-4 recommendations
   │
   └─ OUTPUT FORMAT:
      {
        "strengths": ["Attendance", "Participation"],
        "weak_areas": ["Endurance", "Flexibility"],
        "recommendations": [
          "AI-generated FITT tip",
          "Static evidence-based tip"
        ]
      }
```

---

## Example Prediction Walkthrough

### Input:
```python
{
    "attendance": 85,
    "endurance": 72,
    "strength": 68,
    "flexibility": 60,
    "participation": 90,
    "skill_speed": 70,
    "physical_progress": 75,
    "motivation": 7,
    "stress_level": 4,
    "self_confidence": 7,
    "focus": 6,
    "teamwork": 7,
    "peer_support": 7,
    "communication": 6,
    "sleep_quality": 7,
    "nutrition": 6,
    "screen_time": 4
}
```

### Processing Step-by-Step:

```
1. MIN-MAX SCALING (divide by feature range)
   attendance: 85/100 = 0.850
   endurance: 72/95 = 0.758
   ... (all 17 features scaled)
   
   X = [0.850, 0.758, 0.686, 0.600, 0.818, 0.737, 0.789, 0.64, 0.44, 0.64, 0.60, 0.64, 0.64, 0.60, 0.64, 0.60, 0.44]

2. FORWARD PASS (6 Models)
   
   BPNN:
   hidden = ReLU([X] @ W1 + b1)       → [0.123, 0.456, ...]  (16 neurons)
   output = hidden @ W2 + b2          → 0.748                  (normalized)
   score_bpnn = 0.748 × 100 = 74.8
   
   Random Forest:
   score_rf = rf_model.predict([X])   → 73.2
   
   Gradient Boosting:
   score_gb = gb_model.predict([X])   → 75.1
   
   SVR:
   score_svr = svr_model.predict([X]) → 74.5
   
   Linear Regression:
   score_lr = lr_model.predict([X])   → 75.3
   
   XGBoost:
   score_xgb = xgb_model.predict([X]) → 74.0

3. WEIGHTED ENSEMBLE
   Weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
   
   Final = 0.2×74.8 + 0.2×73.2 + 0.2×75.1 + 0.2×74.5 + 0.2×75.3 + 0.2×74.0
         = 14.96 + 14.64 + 15.02 + 14.90 + 15.06 + 14.80
         = 89.38 ÷ 1.2 = 74.5  (normalized)
         
   Final PE Score: 74.5 / 100

4. SHAP EXPLAINABILITY
   Features ranked by impact:
   ├─ attendance: +2.3 (positive contribution)
   ├─ participation: +1.8
   ├─ physical_progress: +1.5
   ├─ strength: +1.2
   ├─ focus: -0.3
   ├─ stress_level: -0.2
   └─ screen_time: -0.1

5. RECOMMENDATIONS
   Weak Areas: [flexibility, motivation]
   
   AI Coach:
   ├─ "Perform daily 10-minute dynamic stretching routine"
   └─ "Set fitness goals and track progress weekly"
   
   Static Tips:
   ├─ "Hold stretches 30 seconds, 3x daily"
   └─ "Join a fitness group for accountability"
```

### Output:
```json
{
  "final_score": 74.5,
  "confidence": 0.92,
  "individual_predictions": {
    "bpnn": 74.8,
    "random_forest": 73.2,
    "gradient_boosting": 75.1,
    "svr": 74.5,
    "linear_regression": 75.3,
    "xgboost": 74.0
  },
  "shap_explanation": {
    "feature_importance": [
      {"feature": "attendance", "contribution": 2.3},
      {"feature": "participation", "contribution": 1.8},
      ...
    ]
  },
  "recommendations": {
    "strengths": ["Attendance", "Participation"],
    "weak_areas": ["Flexibility", "Motivation"],
    "tips": [
      "Perform daily 10-minute dynamic stretching routine",
      "Set fitness goals and track progress weekly",
      "Hold stretches 30 seconds, 3x daily",
      "Join a fitness group for accountability"
    ]
  },
  "interpretation": "Your PE score of 74.5 is good. Focus on flexibility and motivation to reach excellence."
}
```

---

## Visualizations Generated

### Location: `Upgraded_model/visualizations/`

```
visualizations/
├── step3/                              (Feature Analysis)
│   ├── feature_correlation_values.csv  (Correlation data)
│   ├── summary_statistics.csv          (Min, max, mean, std)
│   └── feature_vs_score/              (Scatter plots)
│       ├── attendance_vs_score.png
│       ├── endurance_vs_score.png
│       ├── strength_vs_score.png
│       ├── flexibility_vs_score.png
│       ├── participation_vs_score.png
│       ├── skill_speed_vs_score.png
│       └── physical_progress_vs_score.png
│
└── ensemble/                          (Ensemble Analysis)
    ├── model_comparison.csv           (Performance metrics)
    ├── r2_scores_comparison.png       (Bar chart)
    ├── rmse_comparison.png            (Bar chart)
    ├── prediction_correlation.png     (Heatmap)
    ├── actual_vs_predicted.png        (Scatter)
    ├── residual_plot.png              (Error distribution)
    └── model_agreement_heatmap.png    (Model correlations)
```

---

## Hyperparameter Summary

### BPNN Training:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 0.0005 | Gradient descent step size |
| Epochs | 200 | Training iterations |
| Batch Size | 256 | Samples per gradient update |
| Gradient Clip | 1.0 | Prevent exploding gradients |
| Hidden Neurons | 16 | (Upgraded) vs 12 (Base) |
| Hidden Activation | ReLU | Non-linearity |
| Output Activation | Linear | Regression problem |
| Weight Init | Xavier | sqrt(1/dim) scaling |

### Sklearn Models:
```python
# Random Forest
RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)

# Gradient Boosting
GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# SVM (SVR)
SVR(kernel='rbf', C=100, gamma='scale')

# Linear Regression
LinearRegression()

# XGBoost
XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
```

---

**Generated:** March 10, 2026
