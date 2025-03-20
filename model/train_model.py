import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# Features & Labels
X = data[['value_cpu', 'value_mem']]
y = data['failure'].astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save Model
with open("../model/model.pkl", "wb") as file:
    pickle.dump(model, file)

