
"""# decisiontree"""

# mushroom_decision_tree.py
"""
Mushroom Edibility Classification using Decision Tree
Compares Gini Impurity vs Information Gain (Entropy) splitting criteria
Dataset: 11_mushroom_edibility.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('dataset/11_mushroom_edibility.csv')
print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df['Class'].value_counts())

# ============================================================
# 2. PREPROCESSING
# ============================================================
# Drop SampleID (not a feature)
df = df.drop(columns=['SampleID'])

# Encode all categorical features and target
le_dict = {}
df_encoded = df.copy()

for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

# Features and target
X = df_encoded.drop(columns=['Class'])
y = df_encoded['Class']

feature_names = X.columns.tolist()
class_names = le_dict['Class'].classes_.tolist()
print("\nFeature names:", feature_names)
print("Class names:", class_names)

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================================
# 4. DECISION TREE - GINI IMPURITY
# ============================================================
dt_gini = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

acc_gini = accuracy_score(y_test, y_pred_gini)
print("\n" + "="*60)
print("DECISION TREE - GINI IMPURITY")
print("="*60)
print(f"Accuracy: {acc_gini:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gini, target_names=class_names))

# ============================================================
# 5. DECISION TREE - INFORMATION GAIN (ENTROPY)
# ============================================================
dt_entropy = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)

acc_entropy = accuracy_score(y_test, y_pred_entropy)
print("\n" + "="*60)
print("DECISION TREE - INFORMATION GAIN (ENTROPY)")
print("="*60)
print(f"Accuracy: {acc_entropy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_entropy, target_names=class_names))

# ============================================================
# 6. COMPARISON
# ============================================================
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"Gini Accuracy     : {acc_gini:.4f}")
print(f"Entropy Accuracy  : {acc_entropy:.4f}")
print(f"Difference        : {abs(acc_gini - acc_entropy):.4f}")

# ============================================================
# 7. PLOT DECISION TREES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(26, 12))

plot_tree(dt_gini, ax=axes[0], feature_names=feature_names,
          class_names=class_names, filled=True, rounded=True,
          fontsize=9, proportion=True)
axes[0].set_title(f"Decision Tree - Gini Impurity\nAccuracy: {acc_gini:.4f}",
                  fontsize=15, fontweight='bold')

plot_tree(dt_entropy, ax=axes[1], feature_names=feature_names,
          class_names=class_names, filled=True, rounded=True,
          fontsize=9, proportion=True)
axes[1].set_title(f"Decision Tree - Information Gain (Entropy)\nAccuracy: {acc_entropy:.4f}",
                  fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('mushroom_decision_trees.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nTree plot saved as 'mushroom_decision_trees.png'")

# ============================================================
# 8. CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_gini = confusion_matrix(y_test, y_pred_gini)
ConfusionMatrixDisplay(cm_gini, display_labels=class_names).plot(
    ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f"Confusion Matrix - Gini\nAccuracy: {acc_gini:.4f}")

cm_entropy = confusion_matrix(y_test, y_pred_entropy)
ConfusionMatrixDisplay(cm_entropy, display_labels=class_names).plot(
    ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title(f"Confusion Matrix - Entropy\nAccuracy: {acc_entropy:.4f}")

plt.tight_layout()
plt.savefig('mushroom_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrices saved as 'mushroom_confusion_matrices.png'")

# ============================================================
# 9. FEATURE IMPORTANCE COMPARISON
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

imp_gini = pd.Series(dt_gini.feature_importances_, index=feature_names).sort_values()
imp_entropy = pd.Series(dt_entropy.feature_importances_, index=feature_names).sort_values()

imp_gini.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title("Feature Importance - Gini", fontweight='bold')
axes[0].set_xlabel("Importance")

imp_entropy.plot(kind='barh', ax=axes[1], color='seagreen')
axes[1].set_title("Feature Importance - Entropy", fontweight='bold')
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig('mushroom_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance plot saved as 'mushroom_feature_importance.png'")

# ============================================================
# 10. TREE DEPTH vs ACCURACY
# ============================================================
depths = range(1, 16)
acc_gini_list, acc_entropy_list = [], []

for d in depths:
    m1 = DecisionTreeClassifier(criterion='gini', max_depth=d, random_state=42).fit(X_train, y_train)
    m2 = DecisionTreeClassifier(criterion='entropy', max_depth=d, random_state=42).fit(X_train, y_train)
    acc_gini_list.append(accuracy_score(y_test, m1.predict(X_test)))
    acc_entropy_list.append(accuracy_score(y_test, m2.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(depths, acc_gini_list, 'o-', label='Gini', color='steelblue', linewidth=2)
plt.plot(depths, acc_entropy_list, 's-', label='Entropy', color='seagreen', linewidth=2)
plt.xlabel("Max Depth", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title("Decision Tree Depth vs Accuracy (Mushroom Edibility)", fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('mushroom_depth_vs_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Depth vs accuracy plot saved as 'mushroom_depth_vs_accuracy.png'")
