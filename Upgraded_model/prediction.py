from datetime import datetime
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import pickle
from recommendation_engine import generate_recommendations

# helper to build paths relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
def rel_path(*parts):
    return os.path.join(base_dir, *parts)

# --------------------------------------------------
# Load Ensemble Models
# --------------------------------------------------
print("\n[System] Loading ensemble models...")  # paths handled with rel_path
models_ensemble = {}
ensemble_info = {}

try:
    # Load BPNN
    bpnn_model = np.load(rel_path("saved_model","bpnn_model.npz"), allow_pickle=True)
    W1_bpnn, b1_bpnn = bpnn_model["W1"], bpnn_model["b1"]
    W2_bpnn, b2_bpnn = bpnn_model["W2"], bpnn_model["b2"]
    
    # Load ML models
    with open(rel_path("saved_model","ensemble","random_forest.pkl"), "rb") as f:
        models_ensemble['rf'] = pickle.load(f)
    with open(rel_path("saved_model","ensemble","gradient_boosting.pkl"), "rb") as f:
        models_ensemble['gb'] = pickle.load(f)
    with open(rel_path("saved_model","ensemble","svr.pkl"), "rb") as f:
        models_ensemble['svr'] = pickle.load(f)
    with open(rel_path("saved_model","ensemble","linear_regression.pkl"), "rb") as f:
        models_ensemble['lr'] = pickle.load(f)
    with open(rel_path("saved_model","ensemble","xgboost.pkl"), "rb") as f:
        models_ensemble['xgb'] = pickle.load(f)
    
    # Load ensemble weights
    with open(rel_path("saved_model","ensemble","ensemble_info.pkl"), "rb") as f:
        ensemble_info = pickle.load(f)
    
    print("✓ All ensemble models loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading ensemble models: {e}")
    print("   Please run step5_ensemble_ml_model.py first")
    exit()

# --------------------------------------------------
# Activation & Math
# --------------------------------------------------
def relu(x):
    return np.maximum(0, x)

# Math for the forward pass:
# $$ \text{Hidden} = \text{ReLU}(X \cdot W_1 + b_1) $$
# $$ \text{Output} = \text{Hidden} \cdot W_2 + b_2 $$

# --------------------------------------------------
# Feature configuration
# --------------------------------------------------
compulsory_features = [
    ("attendance", 50, 100), ("endurance", 30, 95), ("strength", 35, 95),
    ("flexibility", 30, 90), ("participation", 55, 100), ("skill_speed", 30, 95),
    ("physical_progress", 40, 95),
]

optional_features = [
    ("motivation", 4, 9), ("stress_level", 2, 8), ("self_confidence", 4, 9),
    ("focus", 3, 9), ("teamwork", 4, 9), ("peer_support", 5, 9),
    ("communication", 4, 9), ("sleep_quality", 4, 9), ("nutrition", 4, 9),
    ("screen_time", 2, 8),
]

# Create a combined list for SHAP feature naming
feature_names = [f[0] for f in compulsory_features] + [f[0] for f in optional_features]

optional_defaults = {f[0]: 6 for f in optional_features}
optional_defaults["stress_level"] = 5
optional_defaults["screen_time"] = 5

# --------------------------------------------------
# Collect inputs
# --------------------------------------------------
inputs = []
user_inputs_dict = {}

# Descriptive mapping for specific attributes
descriptive_options = {
    "stress_level": [
        ("Low (Calm/Relaxed)", 2), 
        ("Moderate (Manageable)", 5), 
        ("High (Overwhelmed/Anxious)", 8)
    ],
    "sleep_quality": [
        ("Poor (Less than 5 hours / Restless)", 4), 
        ("Fair (5-7 hours / Average)", 6), 
        ("Restorative (7-9+ hours / Deep)", 9)
    ],
    "screen_time": [
        ("Mindful (Under 2 hours daily)", 3), 
        ("Moderate (3-5 hours daily)", 5), 
        ("Excessive (6+ hours daily)", 8)
    ],
    "nutrition": [
        ("Poor (Processed foods/Inconsistent)", 4), 
        ("Balanced (Healthy mix/Regular)", 6), 
        ("Optimal (Nutrient-dense/High fuel)", 9)
    ],
    "focus": [
        ("Distracted (Difficulty staying on task)", 4), 
        ("Steady (Functional concentration)", 6), 
        ("Sharp (High flow state)", 9)
    ],
    "communication": [
        ("Passive (Minimal interaction)", 4), 
        ("Clear (Effective exchanges)", 6), 
        ("Proactive (Leadership/High engagement)", 9)
    ],
    "default": [
        ("Developing", 4), 
        ("Average", 6), 
        ("Strong", 9)
    ]
}

print("\n--- ENTER PHYSICAL ATTRIBUTES ---")
for name, min_v, max_v in compulsory_features:
    while True:
        try:
            val = float(input(f"{name.replace('_',' ').title()} ({min_v}-{max_v}): "))
            if min_v <= val <= max_v:
                inputs.append(val)
                user_inputs_dict[name] = val
                break
            print("❌ Out of range.")
        except ValueError:
            print("❌ Invalid input.")

print("\n--- OPTIONAL ATTRIBUTES ---")
for i, (name, _, _) in enumerate(optional_features, 1):
    print(f"{i}. {name.replace('_',' ').title()}")

choice = input("\nSelect numbers (e.g., 1,3,5) or Enter to skip: ").strip()
selected = [int(i.strip()) for i in choice.split(",") if i.strip().isdigit()] if choice else []

for idx, (name, min_v, max_v) in enumerate(optional_features, 1):
    if idx in selected:
        opts = descriptive_options.get(name, descriptive_options["default"])
        print(f"\nHow would you describe your {name.replace('_',' ').title()}?")
        for opt_idx, (label, _) in enumerate(opts, 1):
            print(f"   {opt_idx}. {label}")
        
        while True:
            try:
                pick = int(input(f"Select choice (1-{len(opts)}): "))
                if 1 <= pick <= len(opts):
                    val = opts[pick-1][1]
                    inputs.append(val)
                    user_inputs_dict[name] = val
                    break
                print(f"❌ Please select a number between 1 and {len(opts)}.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    else:
        val = optional_defaults[name]
        inputs.append(val)
        user_inputs_dict[name] = val

# --------------------------------------------------
# Prediction & Interpretability Logic
# --------------------------------------------------
# Preparing data for the model
X = np.array(inputs, dtype=float)
X_norm = X.copy()
X_norm[:7] /= 100.0  # Normalize physical
X_norm[7:] /= 10.0   # Normalize psych/social
X_norm = X_norm.reshape(1, -1)

# Wrapper function for SHAP to interpret
def model_predict(data):
    hidden_layer = relu(np.dot(data, W1_bpnn) + b1_bpnn)
    output_layer = (np.dot(hidden_layer, W2_bpnn) + b2_bpnn) * 100
    return output_layer.flatten()

# Generate predictions from all ensemble models
print("[System] Generating ensemble predictions...")
y_bpnn = model_predict(X_norm)[0]
y_rf = models_ensemble['rf'].predict(X_norm)[0] * 100
y_gb = models_ensemble['gb'].predict(X_norm)[0] * 100
y_svr = models_ensemble['svr'].predict(X_norm)[0] * 100
y_lr = models_ensemble['lr'].predict(X_norm)[0] * 100
y_xgb = models_ensemble['xgb'].predict(X_norm)[0] * 100

# Weighted ensemble prediction
weights = ensemble_info.get('weights', np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
predicted_score = np.clip(
    weights[0] * y_bpnn +
    weights[1] * y_rf +
    weights[2] * y_gb +
    weights[3] * y_svr +
    weights[4] * y_lr +
    weights[5] * y_xgb,
    0, 100
)

print("\n[System] Performing diagnostic analysis (SHAP)...")
explainer = shap.KernelExplainer(model_predict, np.zeros((1, len(feature_names))))
shap_values = explainer.shap_values(X_norm)

# Identify Key Influencers (Professional Terminology)
indexed_shap = list(enumerate(shap_values[0]))
indexed_shap.sort(key=lambda x: x[1], reverse=True)

primary_positive_influencer = feature_names[indexed_shap[0][0]].replace('_', ' ').title()
major_negative_factor = feature_names[indexed_shap[-1][0]].replace('_', ' ').title()

# --------------------------------------------------
# Recommendations
# --------------------------------------------------
print("\n[System] Compiling your personalized coaching plan...")
recs = generate_recommendations(user_inputs_dict, predicted_score)

# --------------------------------------------------
# Mapping for Descriptive Labels
# --------------------------------------------------
def get_label(name, value):
    labels = {
        "stress_level": {2: "Low (Calm)", 5: "Moderate", 8: "High (Overwhelmed)"},
        "sleep_quality": {4: "Poor (<5 hrs)", 6: "Fair (5-7 hrs)", 9: "Restorative (7-9+ hrs)"},
        "screen_time": {3: "Mindful (<2 hrs)", 5: "Moderate (3-5 hrs)", 8: "Excessive (6+ hrs)"},
        "nutrition": {4: "Poor", 6: "Balanced", 9: "Optimal"},
        "focus": {4: "Distracted", 6: "Steady", 9: "Sharp"},
        "communication": {4: "Passive", 6: "Clear", 9: "Proactive"},
        "default": {4: "Developing", 5: "Moderate", 6: "Average", 9: "Strong"}
    }
    category = labels.get(name, labels["default"])
    return category.get(value, str(value))

# --------------------------------------------------
# Final Unified Output (Console)
# --------------------------------------------------
print("\n" + "═" * 55)
print("           STUDENT PERFORMANCE & DIAGNOSTIC REPORT")
print("═" * 55)
print(f" FINAL ASSESSMENT SCORE: {predicted_score:.2f}%")
print("-" * 55)
print(f"🚀 PRIMARY POSITIVE INFLUENCER: {primary_positive_influencer}")
print(f"📉 MAJOR NEGATIVE FACTOR      : {major_negative_factor}")
print("-" * 55)

if recs.get("strengths"):
    print(f"⭐ TOP STRENGTHS: {', '.join(recs['strengths'])}")

if recs.get("weak_areas"):
    print(f"🔍 FOCUS AREAS: {', '.join(recs['weak_areas'])}")

print("\n📊 RECENT BEHAVIORAL SNAPSHOT:")
for key in ["sleep_quality", "screen_time", "stress_level", "nutrition"]:
    val = user_inputs_dict.get(key)
    print(f"   • {key.replace('_',' ').title()}: {get_label(key, val)}")

print("\n📋 RECOMMENDED ACTION STEPS:")
if recs.get("recommendations"):
    unique_recs = list(dict.fromkeys(recs["recommendations"]))
    for tip in unique_recs:
        print(f"   • {tip}")
else:
    print("   • Keep up the great work!")

print("-" * 55)
print("═" * 55)

# --------------------------------------------------
# Save Detailed Report to Text File
# --------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"PE_Diagnostic_Report_{timestamp}.txt"

try:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("      PHYSICAL EDUCATION DIAGNOSTIC ASSESSMENT REPORT\n")
        f.write(f"      Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")

        f.write("--- CORE METRICS & DIAGNOSTICS ---\n")
        f.write(f"FINAL SCORE               : {predicted_score:.2f}%\n")
        f.write(f"Primary Positive Influencer: {primary_positive_influencer}\n")
        f.write(f"Major Negative Factor     : {major_negative_factor}\n")
        f.write("-" * 60 + "\n\n")

        f.write("--- USER INPUT DATA ---\n")
        for key, value in user_inputs_dict.items():
            label = get_label(key, value) if key in [f[0] for f in optional_features] else ""
            display_val = f"{value} ({label})" if label else value
            f.write(f"{key.replace('_', ' ').title():<25}: {display_val}\n")
        
        f.write("\n" + "-"*60 + "\n")
        if recs.get("strengths"):
            f.write(f"TOP STRENGTHS: {', '.join(recs['strengths'])}\n")
        
        if recs.get("weak_areas"):
            f.write(f"FOCUS AREAS  : {', '.join(recs['weak_areas'])}\n")

        f.write("\nRECOMMENDED ACTION PLAN:\n")
        unique_recs = list(dict.fromkeys(recs.get("recommendations", [])))
        for tip in unique_recs:
            f.write(f"• {tip}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("END OF REPORT\n")

    print(f"\n✅ Detailed diagnostic report saved as: {filename}")

except Exception as e:
    print(f"\n❌ Error saving report: {e}")