"""#knn"""

# KNN Classification
# Loan Approval Dataset

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("dataset/07_loan_approval.csv")

print("First 5 rows:")
print(df.head())

# 2. Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# 3. Remove ApplicantID
df = df.drop("ApplicantID", axis=1)

# 4. Encode categorical columns
categorical_columns = [
    "Education",
    "MaritalStatus",
    "PropertyArea",
    "SelfEmployed",
    "LoanApproved"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 5. Separate features and target
X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

# 6. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 7. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Create KNN model
model = KNeighborsClassifier(n_neighbors=5)

# 9. Train model
model.fit(X_train, y_train)

# 10. Prediction
y_pred = model.predict(X_test)

# 11. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# 12. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 13. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
