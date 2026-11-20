# ai-diabetes-prediction-system
Machine Learning-based Diabetes Prediction System using Python and Logistic Regression to analyze patient health parameters and estimate diabetes risk. Includes data preprocessing, feature scaling, model training, performance evaluation, and prediction visualization.
# ILMA WORKSHOP 01
# AI IN HEALTHCARE: BUILD YOUR FIRST DISEASE PREDICTION MODEL
# Project: Diabetes Prediction
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

print("Libraries imported successfully!")


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Dataset URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(url, names=columns)

print("Dataset loaded successfully!")
print("Number of patients:", len(df))


# ============================================================
# 3. DISPLAY FIRST 5 ROWS
# ============================================================

print("\nFirst 5 rows:")
display(df.head())


# ============================================================
# 4. UNDERSTAND THE DATASET
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
display(df.describe())


# ============================================================
# 5. CHECK FOR MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 6. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nDiabetes outcome distribution:")
print(df["Outcome"].value_counts())

print("\n0 = No diabetes")
print("1 = Diabetes")


# Visualization
plt.figure(figsize=(6, 4))

df["Outcome"].value_counts().plot(kind="bar")

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)

plt.show()


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

print("Features:")
print(X.columns.tolist())

print("\nTarget:")
print("Outcome")


# ============================================================
# 8. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 9. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")


# ============================================================
# 10. CREATE MACHINE LEARNING MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("\nModel created:")
print(model)


# ============================================================
# 11. TRAIN THE MODEL
# ============================================================

model.fit(X_train_scaled, y_train)

print("\nModel training completed!")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)

print("\nPredictions generated!")

print("\nFirst 10 predictions:")
print(y_pred[:10])


# ============================================================
# 13. CALCULATE PREDICTION PROBABILITIES
# ============================================================

y_probability = model.predict_proba(X_test_scaled)[:, 1]

print("\nFirst 10 diabetes risk probabilities:")

for i in range(10):
    print(
        f"Patient {i+1}: "
        f"{y_probability[i]*100:.2f}%"
    )


# ============================================================
# 14. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# 15. PRECISION
# ============================================================

precision = precision_score(y_test, y_pred)

print("\nPrecision:")
print(f"{precision * 100:.2f}%")


# ============================================================
# 16. RECALL
# ============================================================

recall = recall_score(y_test, y_pred)

print("\nRecall:")
print(f"{recall * 100:.2f}%")


# ============================================================
# 17. F1 SCORE
# ============================================================

f1 = f1_score(y_test, y_pred)

print("\nF1 Score:")
print(f"{f1 * 100:.2f}%")


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Diabetes",
            "Diabetes"
        ]
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Visualize confusion matrix

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Diabetes",
        "Diabetes"
    ]
)

disp.plot()

plt.title("Diabetes Prediction - Confusion Matrix")

plt.show()


# ============================================================
# 20. ROC-AUC SCORE
# ============================================================

auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC Score:")
print(f"{auc:.3f}")


# ============================================================
# 21. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Diabetes Prediction")

plt.legend()

plt.show()


# ============================================================
# 22. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

feature_importance["Absolute_Importance"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    "Absolute_Importance",
    ascending=False
)

print("\nFeature importance:")
display(feature_importance)


# Visualization

plt.figure(figsize=(9, 5))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Coefficient"]
)

plt.xlabel("Model Coefficient")

plt.ylabel("Feature")

plt.title("Feature Influence on Diabetes Prediction")

plt.gca().invert_yaxis()

plt.show()


# ============================================================
# 23. CREATE A NEW PATIENT
# ============================================================

new_patient = pd.DataFrame({
    "Pregnancies": [2],
    "Glucose": [140],
    "BloodPressure": [80],
    "SkinThickness": [25],
    "Insulin": [100],
    "BMI": [30],
    "DiabetesPedigreeFunction": [0.5],
    "Age": [45]
})

print("\nNew patient:")
display(new_patient)


# ============================================================
# 24. SCALE NEW PATIENT DATA
# ============================================================

new_patient_scaled = scaler.transform(
    new_patient
)


# ============================================================
# 25. PREDICT NEW PATIENT
# ============================================================

prediction = model.predict(
    new_patient_scaled
)

probability = model.predict_proba(
    new_patient_scaled
)[0][1]


# ============================================================
# 26. DISPLAY FINAL RESULT
# ============================================================

print("\n====================================")
print("        PREDICTION RESULT")
print("====================================")

print(
    f"Estimated diabetes probability: "
    f"{probability * 100:.2f}%"
)

if prediction[0] == 1:

    print("Model Prediction: Higher predicted risk")

else:

    print("Model Prediction: Lower predicted risk")

print("====================================")

print(
    "\nDISCLAIMER:"
    "\nThis prediction is for educational purposes only."
    "\nIt is NOT a medical diagnosis."
)
