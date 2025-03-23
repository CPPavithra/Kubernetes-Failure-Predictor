import pandas as pd
import joblib

# Load trained model
model = joblib.load("../models/failure_predictor.pkl")

# Load new data for prediction
df = pd.read_csv("../data/processed_metrics.csv").drop(columns=["label"]).tail(1)

# Make a prediction
prediction = model.predict(df)
print(f"⚠️ Failure Predicted: {'YES' if prediction[0] == 1 else 'NO'}")

