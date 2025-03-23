import pandas as pd
import numpy as np

def fetch_metric(metric_name):
    """ Generate synthetic metric data for Kubernetes failures """
    np.random.seed(42)
    timestamps = pd.date_range(start="2024-01-01", periods=5000, freq="T")  
    data = {
        "timestamp": timestamps,
        metric_name: np.random.rand(len(timestamps)) * 100  #random for now
    }
    return pd.DataFrame(data)
metrics = [
    "cpu_usage", "memory_usage", "container_network_receive_bytes_total",
    "container_network_transmit_bytes_total", "container_fs_usage_bytes",
    "container_restart_count"
]

# Merge all metrics
data = fetch_metric(metrics[0])
for metric in metrics[1:]:
    metric_df = fetch_metric(metric)
    data = pd.merge(data, metric_df, on="timestamp", how="left")
data["target"] = np.random.choice([0, 1], size=len(data), p=[0.9, 0.1])  
data.to_csv("data/merged_data.csv", index=False)
print("Saved as 'data/merged_data.csv'")

