# ==========================================
# Codomax AI/ML Internship - Day 25
# Topic: Feature Engineering
# Author: Akash Kumar Jha
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 70)
print("             FEATURE ENGINEERING - DAY 25")
print("=" * 70)

# Load dataset
df = pd.read_csv("student_data.csv")

print("\nOriginal Dataset")
print(df)

# ------------------------------------------------
# Feature Engineering
# ------------------------------------------------

# Create average academic score
df["Academic_Average"] = (
    df["Previous_Score"] + df["Assignments_Completed"] * 10
) / 2

# Create attendance category
df["Attendance_Rate"] = df["Attendance"] / 100

# Create study efficiency
df["Study_Efficiency"] = (
    df["Previous_Score"] / df["Study_Hours"]
)

print("\nDataset After Feature Engineering")
print(df)

# Features
features = [
    "Study_Hours",
    "Attendance_Rate",
    "Assignments_Completed",
    "Academic_Average",
    "Study_Efficiency"
]

X = df[features]
y = df["Pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Prediction
predictions = model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Feature importance
print("\nFeature Importance")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))

print("\nFeature Engineering Completed Successfully ✅")
print("=" * 70)
