import numpy as np
import pandas as pd
import ast
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

base_dir = os.path.dirname(os.path.abspath(__file__))
def rel_path(*parts):
    return os.path.join(base_dir, *parts)

# --------------------------------------------------
# Load processed test data
# --------------------------------------------------
test_df = pd.read_csv(rel_path("data","test_processed.csv"))

X_test = np.array(test_df["scaled_features"].apply(ast.literal_eval).tolist())
y_test = test_df["score"].values / 100.0  # Normalize to 0-1

# --------------------------------------------------
# Load trained models (BPNN + Ensemble)
# --------------------------------------------------
print("\n[System] Loading BPNN and ensemble models...")

try:
    # Load BPNN
    bpnn_model = np.load(rel_path("saved_model","bpnn_model.npz"), allow_pickle=True)
    W1 = bpnn_model["W1"]
    b1 = bpnn_model["b1"]
    W2 = bpnn_model["W2"]
    b2 = bpnn_model["b2"]
    
    # Load ML models
    rf_model = pickle.load(open(rel_path("saved_model","ensemble","random_forest.pkl"), "rb"))
    gb_model = pickle.load(open(rel_path("saved_model","ensemble","gradient_boosting.pkl"), "rb"))
    svr_model = pickle.load(open(rel_path("saved_model","ensemble","svr.pkl"), "rb"))
    lr_model = pickle.load(open(rel_path("saved_model","ensemble","linear_regression.pkl"), "rb"))
    xgb_model = pickle.load(open(rel_path("saved_model","ensemble","xgboost.pkl"), "rb"))
    
    # Load ensemble weights
    ensemble_info = pickle.load(open(rel_path("saved_model","ensemble","ensemble_info.pkl"), "rb"))
    weights = ensemble_info.get('weights', np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
    
    print("✓ All models loaded successfully\n")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("   Please run step5_ensemble_ml_model.py first")
    exit()

# --------------------------------------------------
# Activation
# --------------------------------------------------
def relu(x):
    return np.maximum(0, x)

# --------------------------------------------------
# Model Predictions
# --------------------------------------------------
print("[System] Generating predictions from all models...\n")

# BPNN predictions
hidden = relu(np.dot(X_test, W1) + b1)
y_test_bpnn = np.dot(hidden, W2) + b2

# Ensemble ML predictions
y_test_rf = rf_model.predict(X_test)
y_test_gb = gb_model.predict(X_test)
y_test_svr = svr_model.predict(X_test)
y_test_lr = lr_model.predict(X_test)
y_test_xgb = xgb_model.predict(X_test)

# Weighted ensemble
y_pred = (weights[0] * y_test_bpnn.flatten() +
          weights[1] * y_test_rf +
          weights[2] * y_test_gb +
          weights[3] * y_test_svr +
          weights[4] * y_test_lr +
          weights[5] * y_test_xgb)

# Convert to 0-100 scale
y_pred = y_pred * 100
y_test = y_test * 100

# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Tolerance-based "accuracy"
tolerance = 5  # ±5 marks
within_tolerance = np.abs(y_test - y_pred) <= tolerance
tolerance_accuracy = np.mean(within_tolerance) * 100

# --------------------------------------------------
# Results
# --------------------------------------------------
print("="*50)
print("ENSEMBLE MODEL PERFORMANCE EVALUATION")
print("="*50)
print(f"\nModel Type: Weighted Ensemble")
print(f"Components: BPNN + Random Forest + Gradient Boosting + SVR + Linear Regression + XGBoost")
print(f"\nTest Set Performance:")
print("-"*50)
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"R²    : {r2:.4f}")
print(f"Accuracy (±{tolerance} marks): {tolerance_accuracy:.2f}%")
print("="*50)
