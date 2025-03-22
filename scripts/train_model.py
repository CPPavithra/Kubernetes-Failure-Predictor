import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("data/merged_data.csv")

# Check if target column exists
if "target" not in data.columns:
    raise KeyError("❌ 'target' column not found in the dataset!")

# Remove non-numeric columns and separate features/target
X = data.drop(columns=["timestamp", "target"], errors="ignore")
y = data["target"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model with class weighting

model = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
cv_acc = np.mean(cross_val_score(model, X_resampled, y_resampled, cv=5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Feature importance
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
top_features = X.columns[sorted_indices]

# Print results
print("\n📊 MODEL PERFORMANCE METRICS")
print("────────────────────────────────")
print(f"🏋️ Training Accuracy: {train_acc:.4f}")
print(f"🛠️ Test Accuracy: {test_acc:.4f}")
print(f"🎯 Cross-Validation Accuracy: {cv_acc:.4f}")

# Print classification report
print("\n📜 Classification Report:\n", classification_report(y_test, y_pred))

# Print confusion matrix
print("\n🖼️ Confusion Matrix:")
print(cm)

# Show top features
print("\n🔍 Top 5 Most Important Features:")
for i in range(min(5, len(top_features))):
    print(f"   {i+1}. {top_features[i]} ({feature_importances[sorted_indices[i]]:.4f})")

# Save trained model
joblib.dump(model, "models/failure_predictor.pkl")
print("\n✅ Model saved successfully!")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

