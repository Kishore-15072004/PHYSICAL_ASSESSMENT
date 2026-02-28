import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
def rel_path(*parts):
    return os.path.join(base_dir, *parts)

# --------------------------------------------------
# Setup
# --------------------------------------------------
os.makedirs("visualizations", exist_ok=True)
np.random.seed(42)

# --------------------------------------------------
# Load data
# --------------------------------------------------
train_df = pd.read_csv(rel_path("data","train_processed.csv"))
test_df = pd.read_csv(rel_path("data","test_processed.csv"))

X_train = np.array(train_df["scaled_features"].apply(ast.literal_eval).tolist())
X_test = np.array(test_df["scaled_features"].apply(ast.literal_eval).tolist())

y_train = train_df["score"].values.reshape(-1, 1) / 100.0
y_test = test_df["score"].values.reshape(-1, 1) / 100.0

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# --------------------------------------------------
# Activation (ReLU)
# --------------------------------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# --------------------------------------------------
# Network architecture
# --------------------------------------------------
input_neurons = X_train.shape[1]  # now 17
hidden_neurons = 16
output_neurons = 1

W1 = np.random.randn(input_neurons, hidden_neurons) * np.sqrt(1 / input_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons) * np.sqrt(1 / hidden_neurons)
b2 = np.zeros((1, output_neurons))

learning_rate = 0.0005
epochs = 200
batch_size = 256
clip_value = 1.0

loss_history = []

# --------------------------------------------------
# Training (Mini-batch BPNN)
# --------------------------------------------------
n_samples = X_train.shape[0]

for epoch in range(epochs):
    indices = np.random.permutation(n_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0

    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # Forward
        z1 = np.dot(X_batch, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = z2

        loss = np.mean((y_batch - y_pred) ** 2)
        epoch_loss += loss

        # Backprop
        error = y_pred - y_batch

        dW2 = np.dot(a1.T, error)
        db2 = np.sum(error, axis=0, keepdims=True)

        da1 = np.dot(error, W2.T)
        dz1 = da1 * relu_derivative(z1)

        dW1 = np.dot(X_batch.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient clipping
        dW1 = np.clip(dW1, -clip_value, clip_value)
        dW2 = np.clip(dW2, -clip_value, clip_value)

        # Update
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    loss_history.append(epoch_loss)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
a1_test = relu(np.dot(X_test, W1) + b1)
y_test_pred = np.dot(a1_test, W2) + b2

mse = np.mean((y_test - y_test_pred) ** 2)
rmse = np.sqrt(mse) * 100

print("Test RMSE:", rmse)

# --------------------------------------------------
# Plots
# --------------------------------------------------
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("BPNN Training Loss (Stable)")
plt.savefig(rel_path("visualizations","bpnn_training_loss.png"))
plt.close()

plt.figure()
plt.scatter(y_test * 100, y_test_pred * 100, alpha=0.4)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted PE Scores")
plt.savefig(rel_path("visualizations","bpnn_actual_vs_predicted.png"))
plt.close()

print("STEP 4 BPNN COMPLETED SUCCESSFULLY")
# --------------------------------------------------
# Save trained BPNN model
# --------------------------------------------------
os.makedirs(rel_path("saved_model"), exist_ok=True)

np.savez(
    rel_path("saved_model","bpnn_model.npz"),
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2,
    input_neurons=input_neurons,
    hidden_neurons=hidden_neurons,
    output_neurons=output_neurons
)

print("BPNN model saved successfully.")
