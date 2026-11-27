# ============================================================
# ANN BINARY CLASSIFICATION FROM SCRATCH
# Diabetes Dataset
#
# Architecture:
#
# Input:   8 neurons
# Hidden:  7 neurons
# Hidden:  7 neurons
# Hidden:  7 neurons
# Output:  2 neurons
#
# Activation:
# Hidden layers -> tanh
# Output layer  -> softmax
#
# Libraries:
# numpy
# pandas
# matplotlib
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/dataset.csv")

print("Original dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. REMOVE ROWS WITH MISSING TARGET
# ============================================================

# Outcome is our target.
# We cannot train on a sample whose correct class is unknown.

df = df.dropna(subset=["Outcome"]).copy()


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Outcome", axis=1).values

y = df["Outcome"].values.astype(int)


print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 5. HANDLE MISSING FEATURE VALUES
# ============================================================

# Calculate median for every feature

medians = np.nanmedian(X, axis=0)

# Find missing positions

rows, cols = np.where(np.isnan(X))

# Replace missing values with corresponding feature median

for row, col in zip(rows, cols):
    X[row, col] = medians[col]


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

np.random.seed(42)

indices = np.arange(len(X))

np.random.shuffle(indices)

split_index = int(0.8 * len(X))

train_indices = indices[:split_index]

test_indices = indices[split_index:]


X_train = X[train_indices]

X_test = X[test_indices]

y_train = y[train_indices]

y_test = y[test_indices]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. STANDARDIZATION
# ============================================================

# IMPORTANT:
# Calculate mean/std ONLY from training data.

mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0)

# Avoid division by zero

std[std == 0] = 1


X_train = (X_train - mean) / std

X_test = (X_test - mean) / std


# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================

# Class:
#
# 0 -> [1, 0] -> No Diabetes
# 1 -> [0, 1] -> Diabetes

y_train_onehot = np.zeros((len(y_train), 2))

y_train_onehot[
    np.arange(len(y_train)),
    y_train
] = 1


y_test_onehot = np.zeros((len(y_test), 2))

y_test_onehot[
    np.arange(len(y_test)),
    y_test
] = 1


# ============================================================
# 9. NETWORK ARCHITECTURE
# ============================================================

input_neurons = 8

hidden1_neurons = 7

hidden2_neurons = 7

hidden3_neurons = 7

output_neurons = 2


print("\nNetwork architecture:")

print(
    input_neurons,
    "->",
    hidden1_neurons,
    "->",
    hidden2_neurons,
    "->",
    hidden3_neurons,
    "->",
    output_neurons
)


# ============================================================
# 10. INITIALIZE WEIGHTS
# ============================================================

np.random.seed(42)


# Input -> Hidden 1

W1 = np.random.randn(
    input_neurons,
    hidden1_neurons
) * np.sqrt(1 / input_neurons)

b1 = np.zeros((1, hidden1_neurons))


# Hidden 1 -> Hidden 2

W2 = np.random.randn(
    hidden1_neurons,
    hidden2_neurons
) * np.sqrt(1 / hidden1_neurons)

b2 = np.zeros((1, hidden2_neurons))


# Hidden 2 -> Hidden 3

W3 = np.random.randn(
    hidden2_neurons,
    hidden3_neurons
) * np.sqrt(1 / hidden2_neurons)

b3 = np.zeros((1, hidden3_neurons))


# Hidden 3 -> Output

W4 = np.random.randn(
    hidden3_neurons,
    output_neurons
) * np.sqrt(1 / hidden3_neurons)

b4 = np.zeros((1, output_neurons))


print("\nWeight shapes:")

print("W1:", W1.shape)

print("W2:", W2.shape)

print("W3:", W3.shape)

print("W4:", W4.shape)


# ============================================================
# 11. TANH ACTIVATION
# ============================================================

def tanh(x):

    return np.tanh(x)


# ============================================================
# 12. TANH DERIVATIVE
# ============================================================

def tanh_derivative(x):

    return 1 - np.tanh(x) ** 2


# ============================================================
# 13. SOFTMAX
# ============================================================

def softmax(x):

    # Numerical stability

    x = x - np.max(
        x,
        axis=1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=1,
        keepdims=True
    )


# ============================================================
# 14. CROSS ENTROPY LOSS
# ============================================================

def cross_entropy(y_true, y_pred):

    # Prevent log(0)

    epsilon = 1e-12

    y_pred = np.clip(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = -np.sum(
        y_true * np.log(y_pred)
    ) / len(y_true)

    return loss


# ============================================================
# 15. FORWARD PROPAGATION
# ============================================================

def forward(X):

    # --------------------------------------------------------
    # Layer 1
    # --------------------------------------------------------

    Z1 = X @ W1 + b1

    A1 = tanh(Z1)


    # --------------------------------------------------------
    # Layer 2
    # --------------------------------------------------------

    Z2 = A1 @ W2 + b2

    A2 = tanh(Z2)


    # --------------------------------------------------------
    # Layer 3
    # --------------------------------------------------------

    Z3 = A2 @ W3 + b3

    A3 = tanh(Z3)


    # --------------------------------------------------------
    # Output layer
    # --------------------------------------------------------

    Z4 = A3 @ W4 + b4

    A4 = softmax(Z4)


    return Z1, A1, Z2, A2, Z3, A3, Z4, A4


# ============================================================
# 16. LEARNING RATE
# ============================================================

learning_rate = 0.01


# ============================================================
# 17. NUMBER OF EPOCHS
# ============================================================

epochs = 5000


# ============================================================
# 18. STORE LOSS
# ============================================================

loss_history = []


# ============================================================
# 19. TRAINING LOOP
# ============================================================

for epoch in range(epochs):


    # ========================================================
    # FORWARD PROPAGATION
    # ========================================================

    Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward(X_train)


    # ========================================================
    # LOSS
    # ========================================================

    loss = cross_entropy(
        y_train_onehot,
        A4
    )

    loss_history.append(loss)


    # ========================================================
    # BACKPROPAGATION
    # ========================================================

    m = len(X_train)


    # --------------------------------------------------------
    # Output layer
    # --------------------------------------------------------

    # For softmax + cross entropy:
    #
    # dZ4 = prediction - actual

    dZ4 = (A4 - y_train_onehot) / m


    dW4 = A3.T @ dZ4

    db4 = np.sum(
        dZ4,
        axis=0,
        keepdims=True
    )


    # --------------------------------------------------------
    # Hidden Layer 3
    # --------------------------------------------------------

    dA3 = dZ4 @ W4.T

    dZ3 = (
        dA3 *
        tanh_derivative(Z3)
    )


    dW3 = A2.T @ dZ3

    db3 = np.sum(
        dZ3,
        axis=0,
        keepdims=True
    )


    # --------------------------------------------------------
    # Hidden Layer 2
    # --------------------------------------------------------

    dA2 = dZ3 @ W3.T

    dZ2 = (
        dA2 *
        tanh_derivative(Z2)
    )


    dW2 = A1.T @ dZ2

    db2 = np.sum(
        dZ2,
        axis=0,
        keepdims=True
    )


    # --------------------------------------------------------
    # Hidden Layer 1
    # --------------------------------------------------------

    dA1 = dZ2 @ W2.T

    dZ1 = (
        dA1 *
        tanh_derivative(Z1)
    )


    dW1 = X_train.T @ dZ1

    db1 = np.sum(
        dZ1,
        axis=0,
        keepdims=True
    )


    # ========================================================
    # UPDATE WEIGHTS
    # ========================================================

    W4 -= learning_rate * dW4

    b4 -= learning_rate * db4


    W3 -= learning_rate * dW3

    b3 -= learning_rate * db3


    W2 -= learning_rate * dW2

    b2 -= learning_rate * db2


    W1 -= learning_rate * dW1

    b1 -= learning_rate * db1


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch:5d} | Loss = {loss:.6f}"
        )


# ============================================================
# 20. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")

plt.ylabel("Cross Entropy Loss")

plt.title("ANN Training Loss")

plt.grid(True)

plt.show()


# ============================================================
# 21. PREDICTION FUNCTION
# ============================================================

def predict(X):

    _, _, _, _, _, _, _, probabilities = forward(X)

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return predictions, probabilities


# ============================================================
# 22. TEST SET PREDICTION
# ============================================================

y_pred, probabilities = predict(X_test)


# ============================================================
# 23. METRICS
# ============================================================

accuracy = np.mean(
    y_pred == y_test
)

TP = np.sum(
    (y_test == 1) &
    (y_pred == 1)
)

TN = np.sum(
    (y_test == 0) &
    (y_pred == 0)
)

FP = np.sum(
    (y_test == 0) &
    (y_pred == 1)
)

FN = np.sum(
    (y_test == 1) &
    (y_pred == 0)
)


precision = TP / (TP + FP + 1e-12)

recall = TP / (TP + FN + 1e-12)

f1 = (
    2 * precision * recall /
    (precision + recall + 1e-12)
)


print("\n")
print("=" * 60)
print("ANN PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    np.array([
        [TN, FP],
        [FN, TP]
    ])
)


# ============================================================
# 25. DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(min(10, len(X_test))):

    print(
        f"Actual: {y_test[i]} | "
        f"Predicted: {y_pred[i]} | "
        f"P(No Diabetes): {probabilities[i,0]:.4f} | "
        f"P(Diabetes): {probabilities[i,1]:.4f}"
    )