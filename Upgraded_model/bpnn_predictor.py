import numpy as np
import pickle
import os

# utility to construct paths relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
def rel_path(*parts):
    return os.path.join(base_dir, *parts)

# ==================================================
# Activation Function
# ==================================================
def relu(x):
    return np.maximum(0, x)


# ==================================================
# Load Ensemble Models
# ==================================================
def load_ensemble_models():
    """Load all ensemble models and weights"""
    models = {}
    
    try:
        # Load BPNN
        bpnn_model = np.load(rel_path("saved_model","bpnn_model.npz"), allow_pickle=True)
        models['bpnn'] = {
            "W1": bpnn_model["W1"],
            "b1": bpnn_model["b1"],
            "W2": bpnn_model["W2"],
            "b2": bpnn_model["b2"]
        }
        
        # Load ML models
        with open(rel_path("saved_model","ensemble","random_forest.pkl"), "rb") as f:
            models['rf'] = pickle.load(f)
        with open(rel_path("saved_model","ensemble","gradient_boosting.pkl"), "rb") as f:
            models['gb'] = pickle.load(f)
        with open(rel_path("saved_model","ensemble","svr.pkl"), "rb") as f:
            models['svr'] = pickle.load(f)
        with open(rel_path("saved_model","ensemble","linear_regression.pkl"), "rb") as f:
            models['lr'] = pickle.load(f)
        with open(rel_path("saved_model","ensemble","xgboost.pkl"), "rb") as f:
            models['xgb'] = pickle.load(f)
        
        # Load ensemble weights
        with open(rel_path("saved_model","ensemble","ensemble_info.pkl"), "rb") as f:
            ensemble_info = pickle.load(f)
        models['ensemble_info'] = ensemble_info
        
        return models
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please run step5_ensemble_ml_model.py first")
        exit()


# ==================================================
# Feature Definitions
# ==================================================

# First 7 → compulsory
compulsory_features = [
    ("attendance", 50, 100),
    ("endurance", 30, 95),
    ("strength", 35, 95),
    ("flexibility", 30, 90),
    ("participation", 55, 100),
    ("skill_speed", 30, 95),
    ("physical_progress", 40, 95),
]

# Optional indicators
optional_features = [
    ("motivation", 1, 10),
    ("stress_level", 1, 10),
    ("self_confidence", 1, 10),
    ("focus", 1, 10),
    ("teamwork", 1, 10),
    ("peer_support", 1, 10),
    ("communication", 1, 10),
    ("sleep_quality", 1, 10),
    ("nutrition", 1, 10),
    ("screen_time", 1, 10),
]

# Neutral defaults (used if ignored)
optional_defaults = {
    "motivation": 6,
    "stress_level": 5,
    "self_confidence": 6,
    "focus": 6,
    "teamwork": 6,
    "peer_support": 6,
    "communication": 6,
    "sleep_quality": 6,
    "nutrition": 6,
    "screen_time": 5,
}


# ==================================================
# Prediction Function (Ensemble)
# ==================================================
def predict_pe_score(models):

    inputs = []

    print("\nENTER COMPULSORY PHYSICAL ATTRIBUTES")
    print("----------------------------------")

    # -------- compulsory inputs --------
    for name, min_v, max_v in compulsory_features:
        while True:
            try:
                val = float(input(f"{name.replace('_',' ').title()} ({min_v}-{max_v}): "))
                if min_v <= val <= max_v:
                    inputs.append(val)
                    break
                else:
                    print("❌ Value out of range.")
            except:
                print("❌ Invalid input.")

    # -------- optional selection --------
    print("\nOPTIONAL ATTRIBUTES (Select what you want to enter)")
    print("--------------------------------------------------")
    print("Enter numbers separated by commas (example: 1,3,5)")
    print("Press ENTER to skip all\n")

    for i, (name, _, _) in enumerate(optional_features, 1):
        print(f"{i}. {name.replace('_',' ').title()}")

    choice = input("\nYour choice: ").strip()

    selected = []
    if choice:
        selected = [int(i.strip()) for i in choice.split(",") if i.strip().isdigit()]

    # -------- optional inputs --------
    for idx, (name, min_v, max_v) in enumerate(optional_features, 1):
        if idx in selected:
            while True:
                try:
                    val = float(input(f"{name.replace('_',' ').title()} ({min_v}-{max_v}): "))
                    if min_v <= val <= max_v:
                        inputs.append(val)
                        break
                    else:
                        print("❌ Value out of range.")
                except:
                    print("❌ Invalid input.")
        else:
            # ignored → neutral value
            inputs.append(optional_defaults[name])

    # -------- normalization --------
    inputs = np.array(inputs, dtype=float)

    inputs[:7] = inputs[:7] / 100.0   # physical
    inputs[7:] = inputs[7:] / 10.0    # psycho-social

    X = inputs.reshape(1, -1)

    # -------- ensemble predictions --------
    bpnn_model = models['bpnn']
    hidden = relu(np.dot(X, bpnn_model["W1"]) + bpnn_model["b1"])
    y_bpnn = float(np.dot(hidden, bpnn_model["W2"]) + bpnn_model["b2"])[0] * 100
    
    y_rf = float(models['rf'].predict(X)[0]) * 100
    y_gb = float(models['gb'].predict(X)[0]) * 100
    y_svr = float(models['svr'].predict(X)[0]) * 100
    y_lr = float(models['lr'].predict(X)[0]) * 100
    y_xgb = float(models['xgb'].predict(X)[0]) * 100

    # -------- weighted ensemble --------
    weights = models['ensemble_info'].get('weights', np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
    score = (
        weights[0] * y_bpnn +
        weights[1] * y_rf +
        weights[2] * y_gb +
        weights[3] * y_svr +
        weights[4] * y_lr +
        weights[5] * y_xgb
    )
    score = np.clip(score, 0, 100)

    return round(score, 2)


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    print("\n[System] Loading ensemble models...")
    models = load_ensemble_models()
    print("✓ Ensemble models loaded successfully\n")

    pe_score = predict_pe_score(models)

    print("\n----------------------------------")
    print(" Predicted Physical Education Score")
    print(" (Ensemble Model)")
    print("----------------------------------")
    print(f" PE Score: {pe_score}")
    print(f" Model: Weighted Ensemble (BPNN + RF + GB + SVR + LR + XGBoost)")
