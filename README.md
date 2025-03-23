# Kubernetes Failure Prediction

# Deployed Links and Presentation
Check out our Project Report to know more about it-  
GuideWire.pdf
[Download Report PDF](https://drive.google.com/file/d/1ksu7qUSHKJ_9u6-0rzK3GjMCT7YdZFJv/view?usp=sharing)
Check out our video-
[Our Video](https://drive.google.com/file/d/1z3-i6l6DKx3ORYUkF9Mn-4G3l8sQcJBR/view?usp=sharing)

## Index
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
  - [POST /predict](#post-predict)
- [Deployment on Render](#deployment-on-render)
- [Submission Requirements](#submission-requirements)

---

## Project Overview
This project aims to develop a machine learning model to predict failures in Kubernetes clusters based on given or simulated data. The trained model is exposed via a FastAPI service and deployed using Docker and Render.

## Directory Structure
```
.
├── models
│   ├── k8s_failure_model.pkl      # Trained machine learning model
├── scripts
│   ├── train_model.py             # Script for training the model
│   ├── test_model.py              # Script for testing the model
├── app.py                         # FastAPI application
├── Dockerfile                     # Docker configuration
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- Docker
- Render account

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/k8s-failure-prediction.git
   cd k8s-failure-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the FastAPI service locally:
   ```sh
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## Model Training
To train the machine learning model, run:
```sh
python scripts/train_model.py
```
This script loads data, preprocesses it, and trains a classifier to predict Kubernetes failures.

## API Endpoints

### POST /predict
- **Endpoint:** `/predict`
- **Method:** POST
- **Request Body:**
```json
{
    "cpu_usage": 0.5,
    "memory_usage": 0.7,
    "container_network_receive_bytes_total": 3000,
    "container_network_transmit_bytes_total": 2500,
    "container_fs_usage_bytes": 5000,
    "cpu_usage_avg": 0.45,
    "memory_usage_avg": 0.68,
    "container_network_receive_bytes_total_avg": 2900,
    "container_network_transmit_bytes_total_avg": 2400,
    "container_fs_usage_bytes_avg": 4800,
    "container_restart_count_avg": 2
}
```
- **Response:**
```json
{
    "failure_predicted": "YES"
}
```

## Deployment on Render

1. Build and push the Docker image:
   ```sh
   docker build -t your-dockerhub-username/k8s-model:latest .
   docker push your-dockerhub-username/k8s-model:latest
   ```
2. Go to [Render](https://render.com) and create a **new Web Service**.
3. Select **Deploy from Docker** and provide the image name (`your-dockerhub-username/k8s-model:latest`).
4. Set the port to `8000`.
5. Click **Deploy**.
6. Once deployed, test the API using:
   ```sh
   curl -X POST https://your-render-url.onrender.com/predict \
   -H "Content-Type: application/json" \
   -d '{ "cpu_usage": 0.5, "memory_usage": 0.7, ... }'
   ```

## Submission Requirements

- **Model**: A trained machine learning model (`k8s_failure_model.pkl`).
- **Codebase**: Functional code including data collection, model training, and evaluation scripts.
- **Documentation**: Explanation of approach, metrics, and model performance.
- **Presentation**: Recorded demo of the model's predictions and results.
- **Test Data**: Sample data used for testing and validation.

This project follows industry best practices and provides a scalable solution for Kubernetes failure prediction.

