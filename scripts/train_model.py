import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib
import matplotlib.pyplot as plt

# Load Processed Data
df = pd.read_csv("data/processed_metrics.csv")

# Drop unnecessary columns
df = df.drop(columns=["timestamp"], errors="ignore")  

# Ensure "failure" column exists
if "failure" not in df.columns:
    raise ValueError("Error: 'failure' column not found in processed_metrics.csv!")

# Define Features (X) and Target (y)
X = df.drop(columns=["failure"])  
y = df["failure"]  

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save the Model
joblib.dump(model, "models/failure_predictor.pkl")
print("✅ Model saved as models/failure_predictor.pkl")


# Get feature importance
importances = model.feature_importances_
features = X.columns

# Plot
plt.figure(figsize=(10,5))
plt.barh(features, importances, color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.title("Feature Importance in Failure Prediction Model")
plt.show()

# Check training accuracy
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

print(f"🏋️ Training Accuracy: {train_acc:.2f}")
print(f"🛠️ Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
