import requests
import pandas as pd
import os
from datetime import datetime, timezone

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

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
        try:
            timestamp = datetime.fromtimestamp(float(item["value"][0]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            value = float(item["value"][1])
            results.append({"timestamp": timestamp, metric_name: value})
        except Exception as e:
            print(f"❌ Error processing {metric_name}: {e}")

    df = pd.DataFrame(results)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df

# Fetch all metrics
all_data = None

for metric_key, query in METRICS.items():
    df = fetch_metric(query)

    if df.empty:
        print(f"⚠️ Warning: No data for {metric_key}, skipping merge.")
        continue

    if all_data is None:
        all_data = df
    else:
        print(f"Merging {metric_key}...")
        print("Before merge, all_data columns:", list(all_data.columns))
        print("Before merge, df columns:", list(df.columns))

        all_data = pd.merge(all_data, df, on="timestamp", how="outer")

# Save collected data
if all_data is not None and not all_data.empty:
    save_path = os.path.join(SAVE_DIR, "merged_data.csv")
    all_data.to_csv(save_path, index=False)
    print(f"✅ Merged data saved to {save_path}")
    print(all_data.head())  # Preview first few rows
else:
    print("⚠️ No data was fetched, skipping save.")

