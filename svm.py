
"""#svm

"""

# ============================================================
# 1. IMPORT LIBRARIES hu
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from sklearn.decomposition import PCA


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/dataset.csv")

print("Original shape:", df.shape)

print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())


# ============================================================
# 3. REMOVE ROWS WHERE TARGET IS MISSING
# ============================================================

# Outcome is the value we want to predict.
# We cannot train a supervised model without knowing the
# correct class for a training example.

df = df.dropna(subset=["Outcome"]).copy()

print("\nShape after removing missing Outcome:", df.shape)

print("\nMissing values after removing missing Outcome:")
print(df.isnull().sum())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Outcome", axis=1)

y = df["Outcome"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. HANDLE MISSING VALUES IN FEATURES
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)

X_test_imputed = imputer.transform(X_test)


# ============================================================
# 7. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)

X_test_scaled = scaler.transform(X_test_imputed)


# ============================================================
# 8. CREATE SVM
# ============================================================

svm = SVC(
    kernel="linear",
    C=1.0,
    probability=True,
    random_state=42
)


# ============================================================
# 9. TRAIN
# ============================================================

svm.fit(X_train_scaled, y_train)


# ============================================================
# 10. PREDICT
# ============================================================

y_pred = svm.predict(X_test_scaled)

y_probability = svm.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 11. METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_probability)


print("\n============================================================")
print("MODEL PERFORMANCE")
print("============================================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n============================================================")
print("CONFUSION MATRIX")
print("============================================================")

print(cm)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n============================================================")
print("CLASSIFICATION REPORT")
print("============================================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Diabetes", "Diabetes"]
    )
)


# ============================================================
# 14. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["No Diabetes", "Diabetes"]
)

plt.yticks(
    [0, 1],
    ["No Diabetes", "Diabetes"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.show()


# ============================================================
# 15. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"SVM (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 16. SVM WEIGHTS
# ============================================================

weights = pd.DataFrame({
    "Feature": X.columns,
    "Weight": svm.coef_[0]
})

weights["Absolute Weight"] = abs(weights["Weight"])

weights = weights.sort_values(
    "Absolute Weight",
    ascending=False
)

print("\n============================================================")
print("SVM WEIGHTS")
print("============================================================")

print(weights)


# ============================================================
# 17. BIAS
# ============================================================

print("\nSVM Bias:")
print(svm.intercept_[0])


# ============================================================
# 18. SUPPORT VECTORS
# ============================================================

print("\n============================================================")
print("SUPPORT VECTORS")
print("============================================================")

print("Support vectors per class:")
print(svm.n_support_)

print("\nTotal support vectors:")
print(len(svm.support_))


# ============================================================
# 19. PCA: 8D → 2D
# ============================================================

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

print("\n============================================================")
print("PCA")
print("============================================================")

print(
    "PC1 variance:",
    pca.explained_variance_ratio_[0]
)

print(
    "PC2 variance:",
    pca.explained_variance_ratio_[1]
)

print(
    "Total variance:",
    pca.explained_variance_ratio_.sum()
)


# ============================================================
# 20. VISUALIZE CLASSES
# ============================================================

plt.figure(figsize=(9, 7))

for class_value, label in [
    (0, "No Diabetes"),
    (1, "Diabetes")
]:

    mask = y_train.to_numpy() == class_value

    plt.scatter(
        X_train_pca[mask, 0],
        X_train_pca[mask, 1],
        label=label,
        alpha=0.7
    )


# ============================================================
# 21. VISUALIZE SUPPORT VECTORS
# ============================================================

support_vectors_pca = pca.transform(
    svm.support_vectors_
)

plt.scatter(
    support_vectors_pca[:, 0],
    support_vectors_pca[:, 1],
    facecolors="none",
    edgecolors="black",
    s=120,
    linewidths=1.5,
    label="Support Vectors"
)


plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title(
    "Diabetes Dataset — PCA Projection of 8D Feature Space"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 22. SVM DECISION VALUES
# ============================================================

decision_values = svm.decision_function(
    X_test_scaled
)

print("\n============================================================")
print("DECISION FUNCTION")
print("============================================================")

print(decision_values[:10])

