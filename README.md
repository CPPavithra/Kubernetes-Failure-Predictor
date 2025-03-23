# Kubernetes Failure Prediction

## 📌 Project Overview
This project predicts potential failures in Kubernetes clusters using machine learning. The model is trained to detect issues such as:
- 🚨 **Node or pod failures**
- 🖥 **Resource exhaustion** (CPU, memory, disk)
- 🌐 **Network or connectivity issues**
- ⚠️ **Service disruptions** based on logs and events

The solution is packaged into a **FastAPI** service and deployed using **Docker** and **Kubernetes**.

---

## 📂 Directory Structure
```
📦 k8s-failure-prediction
├── 📁 data                      # Raw & processed data files
│   ├── raw_metrics.csv          # Original collected metrics
│   ├── processed_metrics.csv    # Preprocessed data for training
│
├── 📁 models                    # Trained machine learning models
│   ├── failure_predictor.pkl    # Final trained model
│
├── 📁 scripts                   # Model training and evaluation scripts
│   ├── train_model.py           # Script to train the ML model
│   ├── evaluate_model.py        # Model evaluation script
│
├── 📁 app                       # API service
│   ├── app.py                   # FastAPI service for predictions
│   ├── Dockerfile               # Dockerfile for containerization
│
├── 📁 deployment                # Kubernetes deployment files
│   ├── deployment.yaml          # Kubernetes deployment manifest
│   ├── service.yaml             # Kubernetes service manifest
│
├── README.md                    # Documentation
└── requirements.txt              # Python dependencies
```

---

## 🚀 Setup & Installation

### 1️⃣ Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required libraries:
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model
If needed, retrain the model using:
```bash
python scripts/train_model.py
```
The trained model will be saved in the `models/` directory.

### 3️⃣ Run the API Locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Test the API using:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"cpu": 80, "memory": 90, "disk": 70}'
```

---

## 🐳 Dockerization & Kubernetes Deployment

### 🏗️ Build & Run with Docker
1. **Build the Docker image**
```bash
docker build -t pavithra/k8s-failure-predictor:v1 .
```
2. **Run the container**
```bash
docker run -p 8000:8000 pavithra/k8s-failure-predictor:v1
```
3. **Push to Docker Hub**
```bash
docker push pavithra/k8s-failure-predictor:v1
```

### ☸️ Deploy to Kubernetes
1. **Apply deployment and service manifests**
```bash
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
```
2. **Check running pods**
```bash
kubectl get pods
```
3. **Expose the service**
```bash
kubectl port-forward service/k8s-failure-predictor 8000:8000
```
4. **Test the API**
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"cpu": 85, "memory": 95, "disk": 80}'
```

---

## 📊 Model Performance
### ✅ Accuracy Scores:
- **Train Accuracy:** 86.80%
- **Test Accuracy:** 68.88%

### 📉 Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** (No Failure) | 0.82 | 0.64 | 0.72 | 904 |
| **1** (Failure) | 0.56 | 0.77 | 0.65 | 542 |

**Macro Avg:** 69% | **Weighted Avg:** 73%

---

## 📌 Future Improvements
✅ **Enhance Feature Engineering** – Incorporate more time-series trends 📈
✅ **Optimize Hyperparameters** – Use Bayesian optimization 🔬
✅ **Deploy on Cloud** – Host on AWS/GCP/Azure ☁️
✅ **Improve Model Interpretability** – Use SHAP/LIME 📊

---

## 🤝 Contributing
Feel free to fork, contribute, and improve the model. PRs are welcome! 🎯

---

## 🏆 Acknowledgments
Thanks to the open-source community and Kubernetes practitioners for providing valuable datasets and insights!

