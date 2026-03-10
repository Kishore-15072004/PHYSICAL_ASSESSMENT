

---

# 📊 Physical Education Assessment System – Complete Code Analysis (v2.2 Hybrid)

## Project Overview

The **Physical Education Assessment System** is an **AI-driven student performance evaluation platform** that predicts a **Physical Education score (0-100)** using **17 multidimensional attributes**.

The system combines:

* **Deep Learning (BPNN)**
* **Machine Learning (Random Forest + XGBoost)**
* **Explainability (Permutation Feature Importance)**
* **AI Coaching (Groq LLM)**

to provide:

* Accurate PE score prediction
* Feature importance analysis
* Personalized health recommendations.

---

# 🏗 Updated Project Structure

```
Physical_Assessment/

├── base_model/
│
├── Upgraded_model/
│
│   ├── step2_preprocessing_spark.py
│   ├── step3_feature_analysis_spark.py
│   ├── step4_bpnn_model.py
│   ├── step5_hybrid_ml_model.py       ⭐ Updated
│   │
│   ├── bpnn_predictor.py
│   ├── prediction.py
│   ├── recommendation_engine.py
│   ├── model_evaluation.py
│   │
│   ├── demo.py
│   ├── requirements.txt
│   ├── README.md
│   │
│   ├── saved_model/
│   │   ├── bpnn_model.npz
│   │   └── ensemble/
│   │        ├── random_forest.pkl
│   │        ├── xgboost.pkl
│   │        └── hybrid_info.pkl
│
└── data/
```

---

# System Architecture (Updated)

```
Raw Dataset
     │
     ▼
Spark Preprocessing
(17 Features)
     │
     ▼
Feature Analysis
(Correlation + Visualization)
     │
     ▼
Model Training
     │
     ├── BPNN
     ├── Random Forest
     └── XGBoost
     │
     ▼
Hybrid Ensemble
     │
     ▼
Prediction Layer
     │
     ├── PE Score
     ├── PFI Feature Importance
     └── Hybrid Recommendation Engine
             │
             ├── Rule-based Tips
             └── Groq AI Coach
```

---

# 1️⃣ Step 2 — Data Preprocessing (Spark)

File:

```
step2_preprocessing_spark.py
```

### Purpose

Prepare dataset for ML training.

### Operations

1. Load dataset
2. Generate additional behavioral attributes
3. Normalize features
4. Train-test split

---

## Feature Engineering

### Compulsory Physical Metrics (7)

```
attendance
endurance
strength
flexibility
participation
skill_speed
physical_progress
```

### Behavioral + Lifestyle Features (10)

```
motivation
stress_level
self_confidence
focus
teamwork
peer_support
communication
sleep_quality
nutrition
screen_time
```

Example generation:

```
motivation = participation / 10 + random_noise
stress_level = 8 - physical_progress / 15
focus = self_confidence + random_noise
```

---

## Normalization

Min-Max scaling:

```
scaled_value =
(value - min) / (max - min)
```

Output range:

```
0 → 1
```

---

# 2️⃣ Step 3 — Feature Analysis

File

```
step3_feature_analysis_spark.py
```

Purpose:

Statistical understanding of features.

---

## Generated Outputs

```
summary_statistics.csv
feature_correlation_values.csv
scatter_plots/
```

Example correlation results

| Feature     | Correlation |
| ----------- | ----------- |
| attendance  | 0.95        |
| endurance   | 0.92        |
| strength    | 0.90        |
| skill_speed | 0.88        |

---

# 3️⃣ Step 4 — BPNN Deep Learning Model

File

```
step4_bpnn_model.py
```

---

## Neural Network Architecture

```
Input Layer (17 features)

        ↓

Hidden Layer
16 neurons
ReLU activation

        ↓

Output Layer
1 neuron
Linear activation
```

---

## Forward Propagation

```
z1 = X · W1 + b1
a1 = ReLU(z1)

z2 = a1 · W2 + b2
```

Prediction

```
score = z2 * 100
```

---

## Training Parameters

| Parameter     | Value  |
| ------------- | ------ |
| Epochs        | 200    |
| Batch Size    | 256    |
| Learning Rate | 0.0005 |
| Activation    | ReLU   |
| Loss          | MSE    |
| Gradient Clip | 1.0    |

---

# 4️⃣ Step 5 — Hybrid Machine Learning Models

File

```
step5_hybrid_ml_model.py
```

---

## Models Used

### 1️⃣ BPNN

Captures nonlinear relationships.

### 2️⃣ Random Forest

```
RandomForestRegressor(
 n_estimators=100,
 max_depth=15
)
```

Strengths

* Handles nonlinear interactions
* Robust to noise

---

### 3️⃣ XGBoost

```
XGBRegressor(
 n_estimators=100,
 max_depth=5,
 learning_rate=0.1
)
```

Strengths

* Gradient boosted trees
* High predictive power

---

# Hybrid Ensemble Strategy

Predictions from models are combined.

```
Final Score =
0.40 × BPNN
+0.30 × RandomForest
+0.30 × XGBoost
```

Example

```
BPNN = 74.8
RF = 73.5
XGB = 75.1
```

```
Final = 74.5
```

---

# Feature Explainability — PFI (Permutation Feature Importance)

SHAP removed.

Instead using **Permutation Feature Importance**.

---

## Algorithm

```
1 Train model

2 Measure baseline accuracy

3 Shuffle feature column

4 Measure accuracy again

5 Importance =
   accuracy_drop
```

---

## Example PFI Output

| Rank | Feature           | Importance |
| ---- | ----------------- | ---------- |
| 1    | attendance        | 0.31       |
| 2    | endurance         | 0.26       |
| 3    | strength          | 0.22       |
| 4    | participation     | 0.18       |
| 5    | physical_progress | 0.17       |
| 6    | teamwork          | 0.13       |
| 7    | focus             | 0.12       |
| 8    | sleep_quality     | 0.10       |

---

# Prediction Interface

File

```
prediction.py
```

---

## Prediction Pipeline

```
User Inputs
     │
     ▼
Validation
     │
     ▼
Scaling
     │
     ▼
Hybrid Model Prediction
     │
     ▼
PFI Importance Analysis
     │
     ▼
Recommendation Engine
     │
     ▼
Final Report
```

---

# Descriptive Input Options

Example:

```
"stress_level":[
("Low (Calm/Relaxed)",2),
("Moderate (Manageable)",5),
("High (Overwhelmed/Anxious)",8)
]
```

```
"sleep_quality":[
("Poor (Less than 5 hours)",4),
("Fair (5-7 hours)",6),
("Restorative (7-9+ hours)",9)
]
```

---

# Hybrid Recommendation Engine

File

```
recommendation_engine.py
```

---

## Recommendation Layers

```
User Profile Analysis
       │
       ▼
Weak Feature Detection
       │
       ▼
Hybrid Recommendation Engine
```

```
           ┌───────────────┐
           │  Static Tips  │
           └──────┬────────┘
                  │
                  ▼
           ┌───────────────┐
           │ Groq AI Coach │
           └──────┬────────┘
                  │
                  ▼
         Final Recommendations
```

---

# Groq AI Integration

Groq LLM acts as **AI fitness coach**.

---

## Example Prompt

```
Student PE Score: 72
Weak Areas: endurance, flexibility

Provide 3 short actionable fitness recommendations.
```

---

## Example Groq Output

```
1 Jog for 20 minutes three times per week.

2 Perform daily dynamic stretching routine.

3 Join team sports twice weekly.
```

---

# Static Recommendations (Fallback)

Example

```
endurance →
20 minutes jogging 3x weekly
```

```
flexibility →
10 minutes stretching daily
```

```
stress_level →
Practice breathing exercises
```

---

# Model Performance (Updated)

| Model           | R²       | RMSE     |
| --------------- | -------- | -------- |
| BPNN            | 0.812    | 3.01     |
| Random Forest   | 0.79     | 3.15     |
| XGBoost         | 0.80     | 3.10     |
| Hybrid Ensemble | **0.83** | **2.90** |

---

# Example System Output

```
Final PE Score: 74.5
Confidence: High
```

### Model Predictions

```
BPNN: 74.8
Random Forest: 73.5
XGBoost: 75.1
```

---

### Top Influential Features

```
1 Attendance
2 Endurance
3 Strength
4 Participation
```

---

### Strengths

```
Attendance
Participation
Teamwork
```

---

### Weak Areas

```
Flexibility
Endurance
```

---

### Recommendations

```
• Jog 20 minutes three times weekly
• Perform daily dynamic stretching
• Participate in group sports
• Maintain balanced nutrition
```

---

# How to Run the System

### Training

```
python step2_preprocessing_spark.py
python step3_feature_analysis_spark.py
python step4_bpnn_model.py
python step5_hybrid_ml_model.py
```

---

### Prediction

```
python prediction.py
```

---

# Key Advantages of Updated System

### Hybrid ML Architecture

Combines:

```
Deep Learning
+
Tree Ensemble Models
```

---

### Explainability

```
Permutation Feature Importance
```

instead of SHAP.

---

### AI Coaching

Uses **Groq LLM** for:

* personalized fitness advice
* adaptive recommendations

---

### Multidimensional Assessment

Considers:

```
Physical performance
Psychological factors
Lifestyle habits
```

---

# Final System Stack

```
Machine Learning
---------------
BPNN
Random Forest
XGBoost

Explainability
--------------
Permutation Feature Importance

AI Layer
--------
Groq LLM

Recommendation System
---------------------
Hybrid
(Rule-based + AI)
```

---

