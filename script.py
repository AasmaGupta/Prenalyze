import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


plt.ion()

csv_file = "Maternal Health Risk Data Set.csv"

# Load dataset
df = pd.read_csv(csv_file)
print("Dataset loaded")
print(df.head())

# Prepare features and labels
X = df.drop(columns=["RiskLevel"])
y = df["RiskLevel"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix plot
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature importance plot
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
plt.title("Feature Importance for Maternal Risk Prediction")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.show()

# Prediction function
def predict_new_patient(data_dict):
    new_df = pd.DataFrame([data_dict])
    new_scaled = scaler.transform(new_df)
    pred_class = model.predict(new_scaled)[0]
    return le.inverse_transform([pred_class])[0]

# Test prediction
example_patient = {
    "Age": 28,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 10.5,
    "BodyTemp": 98.2,
    "HeartRate": 78
}
print("\nExample patient prediction:", predict_new_patient(example_patient))

# Keep plots open until user closes
input("Press Enter to close plots...")
