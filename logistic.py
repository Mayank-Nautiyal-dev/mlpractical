"""#logistic"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/11_mushroom_edibility.csv")

print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["Class"].value_counts())


# ============================================================
# 2. REMOVE SAMPLE ID
# ============================================================

# SampleID is only an identifier.
# It contains no useful information about mushroom edibility.

df = df.drop("SampleID", axis=1)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X_df = df.drop("Class", axis=1)

y_text = df["Class"]


# ============================================================
# 4. CONVERT TARGET INTO 0 AND 1
# ============================================================

# edible     -> 0
# poisonous  -> 1

y = (y_text == "poisonous").astype(int).values


print("\nTarget encoding:")
print("0 = edible")
print("1 = poisonous")


# ============================================================
# 5. ONE-HOT ENCODE CATEGORICAL FEATURES
# ============================================================

X_df = pd.get_dummies(
    X_df,
    dtype=float
)

X = X_df.values


print("\nNumber of features after one-hot encoding:")
print(X.shape[1])

print("\nEncoded feature names:")
print(X_df.columns.tolist())


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

np.random.seed(42)

indices = np.arange(len(X))

np.random.shuffle(indices)

split = int(0.8 * len(X))

train_indices = indices[:split]
test_indices = indices[split:]


X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. ADD BIAS COLUMN
# ============================================================

# Logistic regression:
#
# z = w1*x1 + w2*x2 + ... + wn*xn + b
#
# We represent b using an additional column containing 1s.

X_train_bias = np.c_[
    np.ones(X_train.shape[0]),
    X_train
]

X_test_bias = np.c_[
    np.ones(X_test.shape[0]),
    X_test
]


print("\nTraining matrix shape:")
print(X_train_bias.shape)


# ============================================================
# 8. INITIALIZE WEIGHTS
# ============================================================

number_of_features = X_train_bias.shape[1]

weights = np.zeros(number_of_features)


# ============================================================
# 9. SIGMOID FUNCTION
# ============================================================

def sigmoid(z):

    # Prevent numerical overflow

    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


# ============================================================
# 10. BINARY CROSS-ENTROPY
# ============================================================

def binary_cross_entropy(y_true, y_pred):

    epsilon = 1e-15

    y_pred = np.clip(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y_true * np.log(y_pred)
        +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


# ============================================================
# 11. TRAIN LOGISTIC REGRESSION
# ============================================================

learning_rate = 0.05

epochs = 5000

loss_history = []

m = len(X_train_bias)


for epoch in range(epochs):

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    # Linear combination

    z = X_train_bias @ weights


    # Convert z into probability

    y_pred = sigmoid(z)


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss = binary_cross_entropy(
        y_train,
        y_pred
    )

    loss_history.append(loss)


    # --------------------------------------------------------
    # GRADIENT
    # --------------------------------------------------------

    gradient = (
        X_train_bias.T @
        (y_pred - y_train)
    ) / m


    # --------------------------------------------------------
    # UPDATE WEIGHTS
    # --------------------------------------------------------

    weights = (
        weights -
        learning_rate * gradient
    )


    if epoch % 500 == 0:

        print(
            "Epoch:",
            epoch,
            "Loss:",
            loss
        )


# ============================================================
# 12. TRAINED WEIGHTS
# ============================================================

print("\nFinal weights:")

for name, weight in zip(
    ["Bias"] + X_df.columns.tolist(),
    weights
):

    print(
        f"{name:30s} {weight:.6f}"
    )


# ============================================================
# 13. PREDICTION
# ============================================================

def predict_probability(X, weights):

    z = X @ weights

    return sigmoid(z)


def predict_class(
    X,
    weights,
    threshold=0.5
):

    probability = predict_probability(
        X,
        weights
    )

    return (
        probability >= threshold
    ).astype(int)


# ============================================================
# 14. TEST PREDICTIONS
# ============================================================

probabilities = predict_probability(
    X_test_bias,
    weights
)

predictions = predict_class(
    X_test_bias,
    weights
)


# ============================================================
# 15. CONFUSION MATRIX VALUES
# ============================================================

TP = np.sum(
    (y_test == 1) &
    (predictions == 1)
)

TN = np.sum(
    (y_test == 0) &
    (predictions == 0)
)

FP = np.sum(
    (y_test == 0) &
    (predictions == 1)
)

FN = np.sum(
    (y_test == 1) &
    (predictions == 0)
)


# ============================================================
# 16. METRICS
# ============================================================

accuracy = (
    (TP + TN) /
    len(y_test)
)

precision = (
    TP / (TP + FP)
    if TP + FP != 0
    else 0
)

recall = (
    TP / (TP + FN)
    if TP + FN != 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall != 0
    else 0
)


print("\n====================================")
print("LOGISTIC REGRESSION RESULTS")
print("====================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix")
print()
print("                 Predicted")
print("              Edible  Poisonous")
print()
print(
    f"Actual Edible    {TN:3d}       {FP:3d}"
)

print(
    f"Actual Poisonous {FN:3d}       {TP:3d}"
)


# ============================================================
# 18. LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")

plt.title(
    "Logistic Regression Training Loss"
)

plt.grid(True)

plt.show()


# ============================================================
# 19. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 5))

plt.scatter(
    range(len(y_test)),
    y_test,
    label="Actual"
)

plt.scatter(
    range(len(predictions)),
    predictions,
    marker="x",
    label="Predicted"
)

plt.xlabel("Test Sample")

plt.ylabel(
    "Class\n0 = Edible, 1 = Poisonous"
)

plt.title(
    "Actual vs Predicted Mushroom Class"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 20. PROBABILITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    probabilities[y_test == 0],
    bins=15,
    alpha=0.7,
    label="Actual Edible"
)

plt.hist(
    probabilities[y_test == 1],
    bins=15,
    alpha=0.7,
    label="Actual Poisonous"
)

plt.axvline(
    0.5,
    linestyle="--",
    label="Threshold = 0.5"
)

plt.xlabel(
    "Predicted Probability of Poisonous"
)

plt.ylabel("Number of Mushrooms")

plt.title(
    "Predicted Probability Distribution"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 21. SAMPLE PREDICTIONS
# ============================================================

print("\nSample Predictions")

for i in range(
    min(15, len(y_test))
):

    actual = (
        "Poisonous"
        if y_test[i] == 1
        else "Edible"
    )

    predicted = (
        "Poisonous"
        if predictions[i] == 1
        else "Edible"
    )

    print(
        f"Actual: {actual:10s} | "
        f"Probability Poisonous: "
        f"{probabilities[i]:.4f} | "
        f"Predicted: {predicted}"
    )
