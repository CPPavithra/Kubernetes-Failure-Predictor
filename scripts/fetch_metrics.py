import requests
import pandas as pd
import os
from datetime import datetime

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

# Define metrics to fetch
METRICS = {
    "cpu_usage": "container_cpu_usage_seconds_total",
    "memory_usage": "container_memory_usage_bytes",
    "disk_io": "node_disk_io_time_seconds_total",
    "network_rx": "node_network_receive_bytes_total",
    "network_tx": "node_network_transmit_bytes_total",
}

SAVE_DIR = "../data"
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_metric(metric_name):
    """Fetches a single metric from Prometheus and returns a DataFrame."""
    response = requests.get(PROMETHEUS_URL, params={"query": metric_name})
    data = response.json()

    results = []
    for item in data.get("data", {}).get("result", []):
        timestamp = datetime.utcfromtimestamp(float(item["value"][0])).strftime("%Y-%m-%d %H:%M:%S")
        value = float(item["value"][1])
        results.append({"timestamp": timestamp, "value": value})

    return pd.DataFrame(results)

# Fetch all metrics
for metric_key, query in METRICS.items():
    df = fetch_metric(query)
    save_path = os.path.join(SAVE_DIR, f"{metric_key}.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ {metric_key} data saved to {save_path}")

