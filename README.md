# Project Name: **AI-Driven Failure Prediction and Analysis System for Kubernetes**

## Overview

This project focuses on **predicting** and **analyzing** failures in Kubernetes clusters using **AI/ML models**. The solution aims to predict potential pod/node failures, resource exhaustion, and network issues based on historical and real-time cluster metrics fetched from **Prometheus**.

Key features include:
- **Data collection**: Historical and real-time metrics are collected from the Kubernetes clusters using Prometheus.
- **Machine Learning Model**: A model is trained to predict failures in the clusters.
- **Evaluation**: The model is evaluated using various metrics to ensure accuracy and reliability.
- **Deployment**: The trained model can be deployed in Kubernetes for real-time prediction.
- **Visualization and Reporting**: Presenting predictions and failure analysis through dashboards and reports.

---

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Model Training](#model-training)
4. [Data Collection](#data-collection)
5. [Fetching Live Prometheus Metrics](#fetching-live-prometheus-metrics)
6. [Model Evaluation](#model-evaluation)
7. [Prediction](#prediction)
8. [Deployment](#deployment)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

---

## Installation

### Prerequisites
- **Python 3.8+** 
- **Kubernetes Cluster** (for deployment)
- **Git** (to clone the repository)
- **Prometheus** (for fetching real-time metrics)

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/kubernetes-failure-prediction.git
cd kubernetes-failure-prediction
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Kubernetes Setup (for deployment)
Follow the instructions in the `deployment/README.md` file to deploy the model into a Kubernetes cluster for real-time prediction.

---

## Project Structure

```plaintext
kubernetes-failure-prediction/
├── data_collection/             # Scripts for collecting data from Kubernetes clusters
│   ├── collect_metrics.py      # Collects metrics from cluster (if applicable)
│   └── preprocess_data.py      # Prepares data for training
├── models/                      # Trained machine learning models
│   ├── failure_predictor.pkl   # Final trained model for failure prediction
├── scripts/                     # Main scripts
│   ├── trainmodel1.py           # Script for training the model
│   ├── fetch_live_metrics.py    # Fetches live metrics from Prometheus
│   ├── predictgemini.py         # Model prediction script
│   └── data_analysis.py         # Data analysis and visualization script
├── deployment/                  # Files for deploying the model to Kubernetes
│   ├── kubernetes_deploy.yaml   # Kubernetes deployment configuration
│   └── Dockerfile               # Dockerfile for containerizing the model
├── tests/                       # Unit and integration tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Model Training

1. **Data Preparation:**
   - Gather data from Kubernetes clusters (e.g., pod status, node status, resource usage metrics).
   - Use `collect_metrics.py` (if applicable) to collect and preprocess the data.

2. **Training the Model:**
   - Ensure your data is clean and formatted correctly.
   - Use `trainmodel1.py` to train the machine learning model. This script will take the preprocessed data and train the model.
   - The trained model will be saved as `failure_predictor.pkl`.

3. **Model Evaluation:**
   - Evaluate the model using accuracy, precision, recall, and F1-score.
   - You can use `evaluate_model.py` to evaluate the model and visualize the results.

---

## Data Collection

This project collects data from Kubernetes clusters using **Prometheus**. You can use `fetch_live_metrics.py` to fetch real-time metrics such as:

- Pod status
- Node status
- Resource usage (CPU, memory, disk, etc.)
- Network traffic and latency
- Error logs and system events

These metrics are then used for training the model and making predictions about potential failures.

---

## Fetching Live Prometheus Metrics

To fetch live metrics from your Kubernetes cluster using Prometheus, use the script:

```bash
python scripts/fetch_live_metrics.py
```

This script connects to Prometheus and retrieves real-time metrics from the Kubernetes cluster. The data fetched will be used as input to the trained model for making real-time failure predictions.

---

## Model Evaluation

After training, the model is evaluated based on its ability to predict failures in the Kubernetes clusters. The following metrics are used to evaluate the model's performance:

- **Accuracy**: Percentage of correct predictions.
- **Precision**: Proportion of positive predictions that are actually correct.
- **Recall**: Proportion of actual positive cases that are correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.

These metrics help in determining the model’s reliability for failure prediction.

---

## Prediction

After training the model and collecting live metrics, you can use the script `predictgemini.py` to make predictions. The model will assess the live metrics and predict potential failures within the Kubernetes cluster.

To make predictions:

```bash
python scripts/predictgemini.py
```

This script will load the trained model (`failure_predictor.pkl`), fetch the live metrics using Prometheus, and output predictions on the potential failures in your cluster.

---

## Deployment

For deploying the trained model into a Kubernetes environment, follow these steps:

1. **Dockerize the model**:
   - Use the `Dockerfile` to create a Docker container for the model.

2. **Kubernetes Deployment**:
   - Use the `kubernetes_deploy.yaml` file to deploy the model into the cluster.

3. **Real-time Prediction**:
   - The model will be deployed to make real-time predictions about potential failures based on live cluster data.

For more detailed instructions, refer to the `deployment/README.md` file.

---

## Usage

### Running the Model Locally

1. **Train the model:**
   Run the following script to train the model:

   ```bash
   python scripts/trainmodel1.py
   ```

2. **Fetch live metrics from Prometheus:**
   To fetch real-time metrics from Prometheus:

   ```bash
   python scripts/fetch_live_metrics.py
   ```

3. **Make predictions:**
   To run predictions using the trained model:

   ```bash
   python scripts/predictgemini.py
   ```

4. **Data Analysis:**
   To perform data analysis and visualize the collected metrics:

   ```bash
   python scripts/data_analysis.py
   ```

---

## Contributing

We welcome contributions! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
