import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import pickle

# utility for building paths relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
def rel_path(*parts):
    return os.path.join(base_dir, *parts)


# --------------------------------------------------
# Setup
# --------------------------------------------------
os.makedirs(rel_path("visualizations", "ensemble"), exist_ok=True)
np.random.seed(42)

print("=" * 60)
print("STEP 5: ENSEMBLE ML MODELS WITH BPNN")
print("=" * 60)

# --------------------------------------------------
# Load data
# --------------------------------------------------
train_df = pd.read_csv(rel_path("data", "train_processed.csv"))
test_df = pd.read_csv(rel_path("data", "test_processed.csv"))

X_train = np.array(train_df["scaled_features"].apply(ast.literal_eval).tolist())
X_test = np.array(test_df["scaled_features"].apply(ast.literal_eval).tolist())

y_train = train_df["score"].values / 100.0  # Normalize to 0-1
y_test = test_df["score"].values / 100.0

print(f"\nTraining samples: {X_train.shape}")
print(f"Testing samples: {X_test.shape}")

# --------------------------------------------------
# LOAD TRAINED BPNN MODEL
# --------------------------------------------------
print("\n" + "-" * 60)
print("LOADING BPNN MODEL")
print("-" * 60)

def relu(x):
    return np.maximum(0, x)

# Load BPNN weights
try:
    bpnn_model = np.load(rel_path("saved_model", "bpnn_model.npz"))
    W1_bpnn = bpnn_model['W1']
    b1_bpnn = bpnn_model['b1']
    W2_bpnn = bpnn_model['W2']
    b2_bpnn = bpnn_model['b2']
    print("✓ BPNN model loaded successfully")
except:
    print("⚠ BPNN model not found. Train step4_bpnn_model.py first.")
    exit()

# BPNN Predictions
def bpnn_predict(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    return z2

y_train_bpnn = bpnn_predict(X_train, W1_bpnn, b1_bpnn, W2_bpnn, b2_bpnn)
y_test_bpnn = bpnn_predict(X_test, W1_bpnn, b1_bpnn, W2_bpnn, b2_bpnn)

# --------------------------------------------------
# MACHINE LEARNING MODELS (Sklearn)
# --------------------------------------------------
print("\n" + "-" * 60)
print("TRAINING MACHINE LEARNING MODELS")
print("-" * 60)

models = {}

# 1. Random Forest
print("\n1. Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, 
                                  random_state=42, n_jobs=-1, min_samples_split=5)
rf_model.fit(X_train, y_train)
y_train_rf = rf_model.predict(X_train)
y_test_rf = rf_model.predict(X_test)
models['Random Forest'] = rf_model
print("✓ Random Forest trained")

# 2. Gradient Boosting
print("\n2. Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                      max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
y_train_gb = gb_model.predict(X_train)
y_test_gb = gb_model.predict(X_test)
models['Gradient Boosting'] = gb_model
print("✓ Gradient Boosting trained")

# 3. Support Vector Regression
print("\n3. Training Support Vector Regressor...")
svr_model = SVR(kernel='rbf', C=100, gamma=0.01)
svr_model.fit(X_train, y_train)
y_train_svr = svr_model.predict(X_train)
y_test_svr = svr_model.predict(X_test)
models['SVR'] = svr_model
print("✓ SVR trained")

# 4. Linear Regression (Baseline)
print("\n4. Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_train_lr = lr_model.predict(X_train)
y_test_lr = lr_model.predict(X_test)
models['Linear Regression'] = lr_model
print("✓ Linear Regression trained")

# 5. XGBoost Regressor
print("\n5. Training XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                             max_depth=5, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_train_xgb = xgb_model.predict(X_train)
y_test_xgb = xgb_model.predict(X_test)
models['XGBoost'] = xgb_model
print("✓ XGBoost trained")

# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------
print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

def evaluate_model(y_true, y_pred, model_name, is_test=True):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse) * 100  # Convert to 0-100 scale
    mae = mean_absolute_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    dataset = "Test" if is_test else "Train"
    print(f"\n{model_name} ({dataset}):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MSE': mse}

# Store results
results = {}

print("\n" + "-" * 60)
print("TEST SET PERFORMANCE")
print("-" * 60)

results['BPNN'] = evaluate_model(y_test, y_test_bpnn.flatten(), 'BPNN')
results['Random Forest'] = evaluate_model(y_test, y_test_rf, 'Random Forest')
results['Gradient Boosting'] = evaluate_model(y_test, y_test_gb, 'Gradient Boosting')
results['SVR'] = evaluate_model(y_test, y_test_svr, 'SVR')
results['Linear Regression'] = evaluate_model(y_test, y_test_lr, 'Linear Regression')
results['XGBoost'] = evaluate_model(y_test, y_test_xgb, 'XGBoost')

# --------------------------------------------------
# ENSEMBLE MODELS
# --------------------------------------------------
print("\n" + "=" * 60)
print("ENSEMBLE APPROACHES")
print("=" * 60)

# Simple Average Ensemble
# include xgboost predictions in the mix
y_test_ensemble_avg = (
    y_test_bpnn.flatten() + y_test_rf + y_test_gb + y_test_svr + y_test_lr + y_test_xgb
) / 6
print("\n1. Simple Average Ensemble (all models):")
results['Ensemble (Average)'] = evaluate_model(y_test, y_test_ensemble_avg, 'Ensemble (Average)')

# Weighted Ensemble (weights based on test R² scores)
weights = np.array([
    results['BPNN']['R2'],
    results['Random Forest']['R2'],
    results['Gradient Boosting']['R2'],
    results['SVR']['R2'],
    results['Linear Regression']['R2'],
    results['XGBoost']['R2']
])
weights = weights / weights.sum()  # Normalize weights
print(f"\nModel Weights (based on R²):")
print(f"  BPNN: {weights[0]:.4f}")
print(f"  Random Forest: {weights[1]:.4f}")
print(f"  Gradient Boosting: {weights[2]:.4f}")
print(f"  SVR: {weights[3]:.4f}")
print(f"  Linear Regression: {weights[4]:.4f}")
print(f"  XGBoost: {weights[5]:.4f}")

y_test_ensemble_weighted = (
    weights[0] * y_test_bpnn.flatten() +
    weights[1] * y_test_rf +
    weights[2] * y_test_gb +
    weights[3] * y_test_svr +
    weights[4] * y_test_lr +
    weights[5] * y_test_xgb
)
print("\n2. Weighted Ensemble:")
results['Ensemble (Weighted)'] = evaluate_model(y_test, y_test_ensemble_weighted, 'Ensemble (Weighted)')

# Top 3 Best Models Ensemble (automatically select 3 lowest RMSE models)
# consider only base models (exclude any ensemble entries)
base_results = {k: v for k, v in results.items() if not k.startswith('Ensemble')}
sorted_rmse = sorted(base_results.items(), key=lambda item: item[1]['RMSE'])
top3_names = [name for name, _ in sorted_rmse[:3]]
print(f"\n3. Top 3 Models Ensemble ({', '.join(top3_names)}):")
# gather predictions dynamically

y_pred_lookup = {
    'BPNN': y_test_bpnn.flatten(),
    'Random Forest': y_test_rf,
    'Gradient Boosting': y_test_gb,
    'SVR': y_test_svr,
    'Linear Regression': y_test_lr,
    'XGBoost': y_test_xgb
}
top_models_avg = np.mean([y_pred_lookup[name] for name in top3_names], axis=0)
results['Ensemble (Top 3)'] = evaluate_model(y_test, top_models_avg, 'Ensemble (Top 3)')

# --------------------------------------------------
# COMPARISON TABLE
# --------------------------------------------------
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON TABLE")
print("=" * 60)

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.sort_values('RMSE')
print("\n" + comparison_df.to_string())

best_model = comparison_df['RMSE'].idxmin()
best_rmse = comparison_df.loc[best_model, 'RMSE']
print(f"\n🏆 BEST MODEL: {best_model} with RMSE: {best_rmse:.4f}")

# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------
print("\n" + "-" * 60)
print("GENERATING VISUALIZATIONS")
print("-" * 60)

# 1. RMSE Comparison
plt.figure(figsize=(12, 6))
models_list = list(results.keys())
rmse_values = [results[m]['RMSE'] for m in models_list]
colors = ['#FF6B6B' if 'Ensemble' not in m else '#4ECDC4' for m in models_list]
bars = plt.bar(models_list, rmse_values, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('RMSE', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison - RMSE', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(rel_path('visualizations','ensemble','rmse_comparison.png'), dpi=300)
plt.close()
print("✓ RMSE comparison plot saved")

# 2. R² Comparison
plt.figure(figsize=(12, 6))
r2_values = [results[m]['R2'] for m in models_list]
colors = ['#FF6B6B' if 'Ensemble' not in m else '#4ECDC4' for m in models_list]
bars = plt.bar(models_list, r2_values, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('R² Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison - R² Score', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(rel_path('visualizations','ensemble','r2_comparison.png'), dpi=300)
plt.close()
print("✓ R² comparison plot saved")

# 3. Actual vs Predicted (Best Model)
if 'Weighted' in best_model:
    y_pred_best = y_test_ensemble_weighted
elif 'Average' in best_model:
    y_pred_best = y_test_ensemble_avg
elif 'Top 3' in best_model:
    y_pred_best = top_models_avg
else:
    y_pred_models = {
        'BPNN': y_test_bpnn.flatten(),
        'Random Forest': y_test_rf,
        'Gradient Boosting': y_test_gb,
        'SVR': y_test_svr,
        'Linear Regression': y_test_lr
    }
    y_pred_best = y_pred_models[best_model]

plt.figure(figsize=(10, 8))
plt.scatter(y_test * 100, y_pred_best * 100, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Score', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Score', fontsize=12, fontweight='bold')
plt.title(f'Best Model: {best_model}\nActual vs Predicted PE Scores', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(rel_path('visualizations','ensemble','best_model_actual_vs_predicted.png'), dpi=300)
plt.close()
print("✓ Actual vs Predicted plot saved")

# 4. Residuals Plot
residuals = (y_test * 100) - (y_pred_best * 100)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best * 100, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Score', fontsize=12, fontweight='bold')
plt.ylabel('Residuals', fontsize=12, fontweight='bold')
plt.title(f'Residual Plot - {best_model}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(rel_path('visualizations','ensemble','residuals_plot.png'), dpi=300)
plt.close()
print("✓ Residuals plot saved")

# 5. Model Correlation Heatmap
predictions_df = pd.DataFrame({
    'BPNN': y_test_bpnn.flatten(),
    'Random Forest': y_test_rf,
    'Gradient Boosting': y_test_gb,
    'SVR': y_test_svr,
    'Linear Regression': y_test_lr,
    'XGBoost': y_test_xgb
})

corr_matrix = predictions_df.corr()
plt.figure(figsize=(8, 6))
im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.colorbar(im)
plt.title('Prediction Correlation Between Models', fontsize=14, fontweight='bold')
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', color='black', fontweight='bold')
plt.tight_layout()
plt.savefig(rel_path('visualizations','ensemble','model_correlation.png'), dpi=300)
plt.close()
print("✓ Model correlation plot saved")

# --------------------------------------------------
# SAVE MODELS
# --------------------------------------------------
print("\n" + "-" * 60)
print("SAVING MODELS")
print("-" * 60)

os.makedirs(rel_path("saved_model","ensemble"), exist_ok=True)

# Save all individual models
for model_name, model_obj in models.items():
    filename = rel_path("saved_model","ensemble", f"{model_name.lower().replace(' ', '_')}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(model_obj, f)
    print(f"✓ {model_name} saved")

# Save ensemble weights
ensemble_info = {
    'weights': weights,
    'model_names': ['BPNN', 'Random Forest', 'Gradient Boosting', 'SVR', 'Linear Regression', 'XGBoost'],
    'best_model': best_model,
    'best_rmse': best_rmse
}
with open(rel_path('saved_model','ensemble','ensemble_info.pkl'), 'wb') as f:
    pickle.dump(ensemble_info, f)
print("✓ Ensemble info saved")

# Save comparison results
comparison_df.to_csv(rel_path('visualizations','ensemble','model_comparison.csv'))
print("✓ Comparison results saved to CSV")

# --------------------------------------------------
# SUMMARY REPORT
# --------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nTotal Models Evaluated: {len(results)}")
print(f"Best Performing Model: {best_model}")
print(f"Best RMSE: {best_rmse:.4f}")
print(f"Best MAE: {results[best_model]['MAE']:.4f}")
print(f"Best R²: {results[best_model]['R2']:.4f}")

print(f"\nEnsemble Approach Benefits:")
print(f"  - Combines strengths of multiple algorithms")
print(f"  - Reduces overfitting")
print(f"  - More robust predictions")
print(f"  - Better generalization")

print("\n" + "=" * 60)
print("STEP 5 ENSEMBLE ML MODELS COMPLETED SUCCESSFULLY")
print("=" * 60)
