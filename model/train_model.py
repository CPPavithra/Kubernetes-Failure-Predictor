import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle

# Load Data
cpu_usage = pd.read_csv("../data/cpu_usage.csv")
memory_usage = pd.read_csv("../data/memory_usage.csv")

# Merge Data
data = cpu_usage.merge(memory_usage, on="timestamp", suffixes=('_cpu', '_mem'))
data = data.drop(columns=["timestamp"])

# Label Failures (1 if CPU > 90% or Memory > 80%)
data['failure'] = (data['value_cpu'] > 0.9) | (data['value_mem'] > 80)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Select only the most important features
important_features = np.argsort(importances)[-10:]  # Keep top 10 features
X_train = X_train.iloc[:, important_features]
X_test = X_test.iloc[:, important_features]

# Features & Labels
X = data[['value_cpu', 'value_mem']]
y = data['failure'].astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,   # Increase trees for stability
    max_depth=10,       # Limit tree depth to reduce complexity
    min_samples_split=10,  # Minimum samples required to split an internal node
    min_samples_leaf=5,   # Minimum samples per leaf to prevent small splits
    max_features="sqrt",  # Use sqrt of features to reduce correlation
    random_state=42
)
model.fit(X_train, y_train)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"✅ Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

# Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save Model
with open("../model/model.pkl", "wb") as file:
    pickle.dump(model, file)

