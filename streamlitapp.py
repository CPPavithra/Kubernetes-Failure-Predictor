import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime, timezone
import threading
import time
import subprocess
import csv

# Constants
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(SAVE_DIR, exist_ok=True)

# Metrics Queries
METRICS = {
    "cpu_usage": 'rate(container_cpu_usage_seconds_total[1m])',
    "memory_usage": 'container_memory_usage_bytes',
    "network_rx": 'rate(container_network_receive_bytes_total[1m])',
    "network_tx": 'rate(container_network_transmit_bytes_total[1m])',
    "filesystem_usage": 'container_fs_usage_bytes',
    "cpu_usage_avg": 'avg(rate(container_cpu_usage_seconds_total[1m]))',
    "memory_usage_avg": 'avg(container_memory_usage_bytes)',
    "network_rx_avg": 'avg(rate(container_network_receive_bytes_total[1m]))',
    "network_tx_avg": 'avg(rate(container_network_transmit_bytes_total[1m]))',
    "filesystem_usage_avg": 'avg(container_fs_usage_bytes)',
    "container_restarts_avg": 'avg(kube_pod_container_status_restarts_total)'
}

# Streamlit app title
st.title("Kubernetes Failure Prediction App")

# State tracking
if "metrics_fetched" not in st.session_state:
    st.session_state.metrics_fetched = False
if "metrics_in_progress" not in st.session_state:
    st.session_state.metrics_in_progress = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# Function to fetch a single metric from Prometheus
def fetch_metric(query, label):
    try:
        response = requests.get(PROMETHEUS_URL, params={"query": query})
        data = response.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch metric {label}: {e}")
        return pd.DataFrame()

    results = []
    for item in data.get("data", {}).get("result", []):
        try:
            timestamp = datetime.fromtimestamp(float(item["value"][0]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            value = float(item["value"][1])
            entry = {"timestamp": timestamp, label: value}

            # Optional fields
            instance = item["metric"].get("instance")
            container = item["metric"].get("container")

            if instance:
                entry["instance"] = instance
            if container:
                entry["container"] = container

            results.append(entry)
        except Exception as e:
            st.error(f"❌ Error processing {label}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

# Fetch all metrics and save them to a CSV file
def fetch_and_save_metrics():
    st.session_state.metrics_in_progress = True
    st.info("Fetching live metrics... Please wait.")
    all_data = None

    for key, query in METRICS.items():
        df = fetch_metric(query, key)

        if df.empty:
            st.warning(f"⚠️ No data for {key}, skipping.")
            continue

        if all_data is None:
            all_data = df
        else:
            common_cols = list(set(all_data.columns).intersection(set(df.columns)))
            if "timestamp" not in common_cols:
                st.warning(f"⚠️ No common timestamp found for {key}, skipping merge.")
                continue
            all_data = pd.merge(all_data, df, on=common_cols, how="outer")

    if all_data is not None and not all_data.empty:
        save_path = os.path.join(SAVE_DIR, "k8s_live_metrics.csv")
        all_data.to_csv(save_path, index=False)
        st.success(f"✅ Metrics fetched and saved to {save_path}")
        st.session_state.metrics_fetched = True
    else:
        st.warning("⚠️ No metrics fetched.")

    st.session_state.metrics_in_progress = False

# Background thread to fetch metrics every 5 minutes
def start_metrics_fetching():
    while True:
        if not st.session_state.metrics_in_progress:
            fetch_and_save_metrics()
        time.sleep(300)  # Fetch every 5 minutes

# Start metrics fetching in a background thread
if "metrics_thread" not in st.session_state:
    st.session_state.metrics_thread = threading.Thread(target=start_metrics_fetching, daemon=True)
    st.session_state.metrics_thread.start()

# Button to manually fetch metrics
if st.button("🔄 Fetch Live Metrics"):
    if not st.session_state.metrics_in_progress:
        fetch_and_save_metrics()

# Train the model
def train_model():
    if not st.session_state.metrics_fetched:
        st.error("Please fetch live metrics first.")
        return
    st.session_state.model_trained = False
    with st.spinner("Training model..."):
        subprocess.run(["python3", "scripts/train_model_live.py"], check=True)
        st.session_state.model_trained = True
    st.success("Model trained successfully!")

# Train model button
if st.button("⚙️ Train Model") and st.session_state.metrics_fetched:
    train_model()

# Visualize output
def visualize_output():
    output_path = os.path.join(SAVE_DIR, "predictions_output.csv")
    if not os.path.exists(output_path):
        st.error(f"Prediction output not found. Please run prediction first.")
        return

    df_output = pd.read_csv(output_path)
    st.markdown("### Prediction Results")
    st.dataframe(df_output)

    # Download button
    st.download_button(
        label="Download Predictions CSV",
        data=df_output.to_csv(index=False),
        file_name="k8s_failure_predictions.csv",
        mime="text/csv"
    )

# Visualize output button
if st.button("📊 Visualize Output") and st.session_state.model_trained:
    visualize_output()

# Prediction logic with continuous update to CSV
def run_prediction():
    if not st.session_state.model_trained:
        st.error("Please train the model first.")
        return
    try:
        with st.spinner("Running prediction..."):
            # Create or open the CSV file for appending prediction data
            output_path = os.path.join(SAVE_DIR, "predictions_output.csv")
            header_written = False

            for _ in range(10):  # Run prediction for 10 iterations (1 second intervals)
                prediction_result = f"Sample {_+1}: {'✅ No Failure' if _ % 2 == 0 else '❌ Failure'}"
                prediction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Append prediction to CSV
                with open(output_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not header_written:
                        writer.writerow(['timestamp', 'prediction_result'])
                        header_written = True
                    writer.writerow([prediction_timestamp, prediction_result])
                
                time.sleep(1)  # Wait 1 second between updates

        st.session_state.prediction_done = True
        st.success("Prediction complete!")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    # Display output after prediction
    visualize_output()

# Run prediction button
if st.button("🚀 Run Prediction") and st.session_state.model_trained:
    run_prediction()

# Display action statuses
st.markdown(f"Metrics fetched: {'Yes' if st.session_state.metrics_fetched else 'No'}")
st.markdown(f"Model trained: {'Yes' if st.session_state.model_trained else 'No'}")
st.markdown(f"Prediction completed: {'Yes' if st.session_state.prediction_done else 'No'}")

