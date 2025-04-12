import pandas as pd
import joblib
import numpy as np

# Load the trained model
MODEL_PATH = "../models/k8s_failure_model_live.pkl"
model = joblib.load(MODEL_PATH)

# CSV path for live metrics (the new data to predict)
CSV_PATH = "/home/pavithra/k8s-failure-prediction/data/k8s_live_metrics.csv"
df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True).str.lower()

# Convert timestamp and set index
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Compute rolling averages only on numeric columns (if not already present)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if f"{col}_avg" not in df.columns:
        df[f"{col}_avg"] = df[col].rolling(window=5, min_periods=1).mean()

# *** Custom Logic for 'target' ***
# Define custom conditions for failure prediction based on metrics such as CPU, Memory, and Restart Counts
cpu_threshold = 0.8  # 80% CPU usage
memory_threshold = 100000000  # 100MB memory usage
restart_threshold = 3  # More than 3 restarts indicating failure

df['cpu_failure'] = df['cpu_usage'].rolling(window=2).apply(lambda x: np.any(x > cpu_threshold), raw=True).fillna(False)
df['memory_failure'] = df['memory_usage'].rolling(window=2).apply(lambda x: np.any(x > memory_threshold), raw=True).fillna(False)
df['restart_failure'] = df['container_restarts_avg'].rolling(window=2).apply(lambda x: np.any(x > restart_threshold), raw=True).fillna(False)

# Combine these conditions to form the target variable (1: Failure, 0: Normal)
df['target'] = ((df['cpu_failure'] > 0) | (df['memory_failure'] > 0) | (df['restart_failure'] > 0)).astype(int)

# Drop non-numeric columns like 'instance' and any rows that are completely empty
df = df.select_dtypes(include=[np.number])

# Handle missing values (NaNs) by imputation (using mean imputation)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")  # Use mean imputation
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separate features (X) and target (y)
X = df_imputed.drop(columns=["target"])

# Make predictions using the trained model
predictions = model.predict(X)

# Print the results
for i, prediction in enumerate(predictions):
    print(f"⚠️ Prediction for sample {i+1}: {'Failure' if prediction == 1 else 'No Failure'}")

