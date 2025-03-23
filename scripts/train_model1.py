import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier

CSV_PATH = "/home/pavithra/k8s-failure-prediction/data/merged_data.csv"
df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True).str.lower()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

for col in df.columns:
    df[f"{col}_avg"] = df[col].rolling(window=5, min_periods=1).mean()
df["target"] = (df["container_restart_count"].diff().fillna(0) > 1).astype(int)
df.drop(columns=["container_restart_count"], inplace=True)

X = df.drop(columns=["target"])
y = df["target"]
# to handle the imbalance
if y.value_counts().min() >= 5:
    smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    X_resampled, y_resampled = X, y

#splitting
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#to reduce overfitting
rf = RandomForestClassifier(
    n_estimators=300,  
    max_depth=10,  
    min_samples_split=20,
    min_samples_leaf=10, 
    bootstrap=True,
    random_state=42
)

xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

#predict
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

#combining them
y_pred_ensemble = (y_pred_rf + y_pred_xgb) // 2
train_acc = rf.score(X_train, y_train) * 100
test_acc = accuracy_score(y_test, y_pred_ensemble) * 100
print(f"\n🎯 Train Accuracy: {train_acc:.2f} %")
print(f"🎯 Test Accuracy: {test_acc:.2f} %")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred_ensemble))


MODEL_PATH = "../models/k8s_failure_model.pkl"
joblib.dump(rf, MODEL_PATH)
model = joblib.load("models/k8s_failure_model.pkl")
print("The features in model are\n")
print(model.feature_names_in_)
print(f"\n✅ Model saved at {MODEL_PATH}")

#confusion matrix to be plotted
cm = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#feature importance to be plotted
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False).head(15)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette="viridis")
plt.title("Top 15 Important Features")
plt.show()

