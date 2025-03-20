import os
import requests
import pandas as pd
from datetime import datetime

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

def fetch_metric(metric_query, metric_name):
    """Fetch metrics from Prometheus with error handling."""
    try:
        response = requests.get(PROMETHEUS_URL, params={'query': metric_query}, timeout=5)
        response.raise_for_status()  # Raise an error if request fails
        data = response.json()

        if 'data' not in data or 'result' not in data['data']:
            print(f"⚠️ No data found for {metric_name}")
            return pd.DataFrame(columns=['timestamp', metric_name])  # Empty DataFrame

        results = []
        for item in data['data']['result']:
            try:
                timestamp = datetime.utcfromtimestamp(float(item['value'][0])).strftime('%Y-%m-%d %H:%M:%S')
                value = float(item['value'][1])
                results.append({'timestamp': timestamp, metric_name: value})
            except (ValueError, IndexError):
                print(f"⚠️ Skipping invalid data point in {metric_name}: {item}")

        return pd.DataFrame(results)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching {metric_name}: {e}")
        return pd.DataFrame(columns=['timestamp', metric_name])

# Ensure the 'data' directory exists
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
os.makedirs(output_dir, exist_ok=True)

# Fetch Metrics with correct queries
cpu_usage = fetch_metric('rate(container_cpu_usage_seconds_total[1m])', 'cpu_usage')  # CPU as rate
memory_usage = fetch_metric('container_memory_usage_bytes', 'memory_usage')  # Memory in bytes

# Convert Memory Usage to MB
if not memory_usage.empty:
    memory_usage['memory_usage'] = memory_usage['memory_usage'] / (1024 * 1024)  # Convert to MB

# Save to CSV if data exists
if not cpu_usage.empty:
    cpu_usage.to_csv(os.path.join(output_dir, "cpu_usage.csv"), index=False)
    print("✅ CPU usage saved to data/cpu_usage.csv")
else:
    print("⚠️ No CPU usage data to save.")

if not memory_usage.empty:
    memory_usage.to_csv(os.path.join(output_dir, "memory_usage.csv"), index=False)
    print("✅ Memory usage saved to data/memory_usage.csv")
else:
    print("⚠️ No memory usage data to save.")

