
---

# 🎨 Visual Architecture & Component Summary (Hybrid + Groq + PFI)

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│          PHYSICAL EDUCATION ASSESSMENT SYSTEM v2.1              │
│        Hybrid ML + AI Recommendation Architecture               │
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
    ┌──────────────────────┐     ┌──────────────────────┐
    │ STEP 3: ANALYSIS     │     │ STEP 4: MODEL TRAIN  │
    │ - Correlation study  │     │ - BPNN training      │
    │ - Stat summaries     │     │ - 17 input neurons   │
    │ - Visualization      │     │ - 16 hidden neurons  │
    └──────────────────────┘     │ - Backpropagation    │
                                 └──────────┬───────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │ BPNN Weights     │
                                   │ (bpnn_model.npz) │
                                   └────────┬─────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │ STEP 5: ML MODELS       │
                              │                         │
                              │ ├─ Random Forest        │
                              │ └─ XGBoost              │
                              │                         │
                              └──────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────────┐
                          │ HYBRID ENSEMBLE PREDICTION      │
                          │                                 │
                          │  BPNN           → weight 0.40   │
                          │  Random Forest  → weight 0.30   │
                          │  XGBoost        → weight 0.30   │
                          │                                 │
                          │ Final Score = Weighted Average  │
                          └───────────────┬─────────────────┘
                                          │
                                          ▼
                         ┌────────────────────────────────┐
                         │ SAVED MODELS                   │
                         │ saved_model/ensemble/          │
                         │                                │
                         │ ├─ random_forest.pkl           │
                         │ ├─ xgboost.pkl                 │
                         │ ├─ bpnn_model.npz              │
                         │ └─ ensemble_info.pkl           │
                         └─────────────┬──────────────────┘
                                       │
                                       ▼
                      ┌──────────────────────────────────┐
                      │  INFERENCE LAYER (prediction.py) │
                      │                                  │
                      │ 1. User Input Collection         │
                      │ 2. Min-Max Scaling               │
                      │ 3. Hybrid Model Prediction       │
                      │ 4. Weighted Ensemble             │
                      │ 5. PFI Feature Importance        │
                      │ 6. Hybrid Recommendation Engine  │
                      └────────────┬─────────────────────┘
                                   │
           ┌───────────────────────┼────────────────────────┐
           │                       │                        │
           ▼                       ▼                        ▼
┌─────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ Final PE Score      │  │ Feature Importance   │  │ AI Recommendations   │
│ (0-100 scale)       │  │ via PFI              │  │ via Groq API         │
│                     │  │                      │  │                      │
│ ✓ Hybrid ensemble   │  │ ✓ Top factors        │  │ ✓ AI fitness coach   │
│ ✓ Confidence score  │  │ ✓ Feature ranking    │  │ ✓ Personalized tips  │
└─────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

---

# Hybrid Model Prediction Flow

```
User Input (17 features)
        │
        ▼
 Min-Max Scaling (0-1)
        │
        ▼
 ┌──────────────────────────────┐
 │ Model Predictions            │
 │                              │
 │ BPNN            → 74.8       │
 │ Random Forest   → 73.5       │
 │ XGBoost         → 75.        │
 └───────────────┬──────────────┘
                 │
                 ▼
        HYBRID ENSEMBLE
```

```
Final Score =
0.40 × BPNN
+0.30 × RandomForest
+0.30 × XGBoost
```

Example:

```
Final =
0.40×74.8 +
0.30×73.5 +
0.30×75.1

= 74.5
```

---

# Feature Importance (PFI instead of SHAP)

### Permutation Feature Importance Process

```
Baseline Prediction Accuracy
        │
        ▼
Shuffle one feature column
        │
        ▼
Recalculate model accuracy
        │
        ▼
Performance Drop =
Feature Importance
```

Example Output

```
Rank │ Feature              │ Importance
─────┼──────────────────────┼────────────
1    │ attendance           │ 0.31
2    │ endurance            │ 0.26
3    │ strength             │ 0.21
4    │ participation        │ 0.18
5    │ physical_progress    │ 0.16
6    │ skill_speed          │ 0.15
7    │ teamwork             │ 0.13
8    │ focus                │ 0.12
9    │ sleep_quality        │ 0.09
10   │ nutrition            │ 0.08
```

---

# Hybrid Recommendation Engine (Groq + Rule System)

### Two Layer Recommendation System

```
Weak Feature Detection
        │
        ▼
Feature Threshold Logic
        │
        ▼
Weak Areas Identified
        │
        ▼
┌─────────────────────────────┐
│ Hybrid Recommendation Engine│
└───────────────┬─────────────┘
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
 Static Rules       Groq AI Coach
 (deterministic)    (LLM reasoning)
        │                │
        │                │
        ▼                ▼
Evidence Tips      Personalized Advice
        │                │
        └───────┬────────┘
                ▼
        Final Recommendations
```

---

# Groq API Usage

### Example Flow

```
User PE Score: 72
Weak Areas: endurance, flexibility
```

Prompt sent to Groq:

```
You are a Physical Education coach.

Student score: 72/100
Weak areas: endurance, flexibility

Provide 3 short actionable fitness tips.
```

Groq Response Example

```
1. Jog 20 minutes 3 times weekly to improve cardiovascular endurance.

2. Perform dynamic stretching for 10 minutes daily to improve flexibility.

3. Include bodyweight circuit training twice weekly.
```

---

# Example Final Output

```
Final PE Score: 74.5
Confidence: High

Model Predictions
------------------
BPNN:           74.8
Random Forest:  73.5
XGBoost:        75.1

Top Influential Features (PFI)
------------------------------
1. Attendance
2. Endurance
3. Strength
4. Participation

Strengths
---------
Attendance
Participation

Weak Areas
----------
Flexibility
Endurance

Recommendations
---------------
• Jog 20 minutes three times weekly
• Perform daily dynamic stretching
• Join team sports to improve stamina
```

---

# Final Model Stack

```
ML Layer
──────────────
BPNN
Random Forest
XGBoost

Explainability
──────────────
Permutation Feature Importance (PFI)

AI Layer
──────────────
Groq LLM (AI Fitness Coach)

Recommendation Layer
──────────────
Hybrid System
(Rules + LLM)
```

---
