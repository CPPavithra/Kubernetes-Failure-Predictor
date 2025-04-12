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
from prometheus_api_client import PrometheusConnect
import datetime

# Connect to Prometheus
prom = PrometheusConnect(url="http://<your-prometheus-server>:9090", disable_ssl=True)

# Fetch live data from Prometheus
def get_prometheus_data(metric_name, start_time, end_time, step="5m"):
    query = f'{metric_name}'
    result = prom.custom_query_range(
        query=query,
        start=start_time,
        end=end_time,
        step=step
    )
    return result

# Preprocess the data into a DataFrame
def preprocess_data_from_prometheus(data):
    # Assuming that the data returned is a list of dictionaries
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Compute rolling averages (you can adjust the window size)
    for col in df.columns:
        df[f"{col}_avg"] = df[col].rolling(window=5, min_periods=1).mean()

    # Creating target variable based on "container_restart_count"
    df["target"] = (df["container_restart_count"].diff().fillna(0) > 1).astype(int)
    df.drop(columns=["container_restart_count"], inplace=True)
    
    return df

# Specify the time range
start_time = datetime.datetime.now() - datetime.timedelta(hours=1)
end_time = datetime.datetime.now()

# Fetch Prometheus data for specific metrics (e.g., 'cpu_usage', 'memory_usage')
cpu_usage_data = get_prometheus_data('cpu_usage', start_time, end_time)
memory_usage_data = get_prometheus_data('memory_usage', start_time, end_time)

# Assuming you are working with CPU usage and Memory usage data
df = preprocess_data_from_prometheus(cpu_usage_data)
memory_df = preprocess_data_from_prometheus(memory_usage_data)

# Merge or concatenate the data as needed
df = pd.merge(df, memory_df, left_index=True, right_index=True, how="inner")

# Now, proceed with the same machine learning pipeline as you had
X = df.drop(columns=["target"])
y = df["target"]

# To handle class imbalance
if y.value_counts().min() >= 5:
    smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    X_resampled, y_resampled = X, y

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the models
rf = RandomForestClassifier(
    n_estimators=300,  
    max_depth=10,  
    min_samples_split=20,
    min_samples_leaf=10, 
    bootstrap=True,
    random_state=42
)

xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42)

# Train the models
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predict using the trained models
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

# Combine the predictions
y_pred_ensemble = (y_pred_rf + y_pred_xgb) // 2

# Calculate accuracies
train_acc = rf.score(X_train, y_train) * 100
test_acc = accuracy_score(y_test, y_pred_ensemble) * 100

print(f"\n🎯 Train Accuracy: {train_acc:.2f} %")
print(f"🎯 Test Accuracy: {test_acc:.2f} %")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred_ensemble))

# Save the model
MODEL_PATH = "../models/k8s_failure_model.pkl"
joblib.dump(rf, MODEL_PATH)
model = joblib.load(MODEL_PATH)
print("The features in model are\n")
print(model.feature_names_in_)
print(f"\n✅ Model saved at {MODEL_PATH}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance plotting
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False).head(15)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette="viridis")
plt.title("Top 15 Important Features")
plt.show()
