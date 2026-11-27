
"""#naivebias"""

# loan_naive_bayes.py
"""
Loan Approval Classification using Naive Bayes
Dataset: 07_loan_approval.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('dataset/07_loan_approval.csv')
print("Dataset shape:", df.shape)
print("\nTarget distribution:\n", df['LoanApproved'].value_counts())

# ============================================================
# 2. PREPROCESSING
# ============================================================
df = df.drop(columns=['ApplicantID'])

# Encode categorical features
cat_cols = ['Education', 'MaritalStatus', 'PropertyArea', 'SelfEmployed']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
df['LoanApproved'] = le_target.fit_transform(df['LoanApproved'])  # N=0, Y=1
print("Target classes:", le_target.classes_)

# Features and target
X = df.drop(columns=['LoanApproved'])
y = df['LoanApproved']
feature_names = X.columns.tolist()
print("Features:", feature_names)

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features (helps GaussianNB)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. NAIVE BAYES MODEL
# ============================================================
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred = nb.predict(X_test_scaled)
y_proba = nb.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
print("\n" + "="*60)
print("NAIVE BAYES - LOAN APPROVAL")
print("="*60)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['N', 'Y']))

# ============================================================
# 5. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['N', 'Y'], yticklabels=['N', 'Y'])
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title(f"Confusion Matrix - Naive Bayes\nAccuracy: {acc:.4f}",
          fontweight='bold')
plt.tight_layout()
plt.savefig('loan_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved as 'loan_confusion_matrix.png'")

# ============================================================
# 6. ROC CURVE
# ============================================================
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Naive Bayes (Loan Approval)', fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loan_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("ROC curve saved as 'loan_roc_curve.png'")

# ============================================================
# 7. FEATURE DISTRIBUTIONS BY CLASS (Naive Bayes assumption viz)
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(22, 10))
axes = axes.flatten()

for i, col in enumerate(feature_names):
    axes[i].hist(X_train[y_train == 0][col], bins=25, alpha=0.6,
                 label='Not Approved (N)', color='crimson', density=True)
    axes[i].hist(X_train[y_train == 1][col], bins=25, alpha=0.6,
                 label='Approved (Y)', color='seagreen', density=True)
    axes[i].set_title(col, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

for j in range(len(feature_names), len(axes)):
    axes[j].axis('off')

plt.suptitle("Feature Distributions by Class - Naive Bayes Assumption Check",
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('loan_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature distributions saved as 'loan_feature_distributions.png'")

# ============================================================
# 8. PRIOR & LEARNED PARAMETERS (Naive Bayes internals)
# ============================================================
print("\n" + "="*60)
print("NAIVE BAYES LEARNED PARAMETERS")
print("="*60)
print("\nClass Priors:")
for cls, prior in zip(le_target.classes_, nb.class_prior_):
    print(f"  P({cls}) = {prior:.4f}")

print("\nClass Means (theta_):")
means_df = pd.DataFrame(nb.theta_, columns=feature_names,
                        index=[f"Class_{c}" for c in le_target.classes_])
print(means_df.round(3))

print("\nClass Variances (sigma_):")
var_df = pd.DataFrame(nb.var_, columns=feature_names,
                      index=[f"Class_{c}" for c in le_target.classes_])
print(var_df.round(3))

# ============================================================
# 9. VISUALIZE LEARNED MEANS
# ============================================================
means_plot = pd.DataFrame(nb.theta_, columns=feature_names,
                          index=['N', 'Y']).T

means_plot.plot(kind='bar', figsize=(14, 6), color=['crimson', 'seagreen'])
plt.title("Naive Bayes Learned Feature Means per Class",
          fontweight='bold', fontsize=13)
plt.ylabel("Mean (standardized scale)")
plt.xlabel("Feature")
plt.xticks(rotation=45, ha='right')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('loan_nb_means.png', dpi=150, bbox_inches='tight')
plt.show()
print("Learned means plot saved as 'loan_nb_means.png'")

# ============================================================
# 10. THRESHOLD ANALYSIS
# ============================================================
thresholds = np.arange(0.1, 0.95, 0.05)
accs, precisions, recalls = [], [], []

for t in thresholds:
    preds = (y_proba >= t).astype(int)
    accs.append(accuracy_score(y_test, preds))
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accs, 'o-', label='Accuracy', linewidth=2)
plt.plot(thresholds, precisions, 's-', label='Precision', linewidth=2)
plt.plot(thresholds, recalls, '^-', label='Recall', linewidth=2)
plt.xlabel("Decision Threshold", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Naive Bayes - Threshold Analysis (Loan Approval)",
          fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loan_threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Threshold analysis saved as 'loan_threshold_analysis.png'")

# ============================================================
# 11. FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY - NAIVE BAYES (LOAN APPROVAL)")
print("="*60)
print(f"Accuracy  : {acc:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"Test Size : {len(y_test)} samples")
print(f"Train Size: {len(y_train)} samples")
