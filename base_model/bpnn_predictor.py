import numpy as np

# -----------------------------
# Activation function
# -----------------------------
def relu(x):
    return np.maximum(0, x)

# -----------------------------
# Load saved BPNN model
# -----------------------------
def load_bpnn_model(model_path="saved_model/bpnn_model.npz"):
    model = np.load(model_path, allow_pickle=True)

    return {
        "W1": model["W1"],
        "b1": model["b1"],
        "W2": model["W2"],
        "b2": model["b2"]
    }

# -----------------------------
# Min-Max Scaling (same logic as training)
# -----------------------------
def min_max_scale(values, min_val=0, max_val=100):
    values = np.array(values, dtype=float)
    return (values - min_val) / (max_val - min_val)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_pe_score(
    attendance,
    endurance,
    strength,
    flexibility,
    participation,
    skill_speed,
    physical_progress,
    model
):
    # Step 1: Create feature vector
    features = [
        attendance,
        endurance,
        strength,
        flexibility,
        participation,
        skill_speed,
        physical_progress
    ]

    # Step 2: Scale features
    X = min_max_scale(features).reshape(1, -1)

    # Step 3: Forward pass
    z1 = np.dot(X, model["W1"]) + model["b1"]
    a1 = relu(z1)

    z2 = np.dot(a1, model["W2"]) + model["b2"]

    # Step 4: Convert to PE score (0–100)
    predicted_score = float(z2[0][0] * 100)
    predicted_score = np.clip(predicted_score, 0, 100)

    return round(predicted_score, 2)

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    model = load_bpnn_model()

    score = predict_pe_score(
        attendance=85,
        endurance=72,
        strength=68,
        flexibility=60,
        participation=90,
        skill_speed=70,
        physical_progress=75,
        model=model
    )

    print("Predicted PE Score:", score)
