import numpy as np
import ast
import pandas as pd

# -----------------------------
# Activation
# -----------------------------
def relu(x):
    return np.maximum(0, x)

# -----------------------------
# Load saved model
# -----------------------------
model = np.load("saved_model/bpnn_model.npz", allow_pickle=True)

W1 = model["W1"]
b1 = model["b1"]
W2 = model["W2"]
b2 = model["b2"]

print("Model loaded successfully.")

# -----------------------------
# Load new data (example)
# -----------------------------
# Example: load test set or new student data
test_df = pd.read_csv("data/test_processed.csv")

X_test = np.array(
    test_df["scaled_features"].apply(ast.literal_eval).tolist()
)

# -----------------------------
# Forward pass (prediction)
# -----------------------------
hidden = relu(np.dot(X_test, W1) + b1)
y_pred = np.dot(hidden, W2) + b2

# Convert back to score scale
predicted_scores = y_pred * 100

print("Sample predictions:")
print(predicted_scores[:10])
