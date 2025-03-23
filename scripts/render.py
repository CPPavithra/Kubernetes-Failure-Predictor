import pickle
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load trained model
model_path = "../models/k8s_failure_model.pkl"  # Change this to your actual model path
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define input data structure
class ModelInput(BaseModel):
    features: list[float]  # Example: [5.1, 3.5, 1.4, 0.2]

# Root endpoint
@app.get("/")
def home():
    return {"message": "K8s Failure Prediction API is Running!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: ModelInput):
    try:
        # Convert input to NumPy array and reshape for prediction
        input_data = np.array(data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        return {"error": str(e)}

# Run the server if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

