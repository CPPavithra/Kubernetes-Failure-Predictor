import pandas as pd
import matplotlib.pyplot as plt

# Load data
cpu_df = pd.read_csv("data/cpu_usage.csv")
memory_df = pd.read_csv("data/memory_usage.csv")

# Convert timestamps to datetime
cpu_df["timestamp"] = pd.to_datetime(cpu_df["timestamp"])
memory_df["timestamp"] = pd.to_datetime(memory_df["timestamp"])

# Convert Memory Usage from bytes to MB
memory_df["memory_usage"] = memory_df["memory_usage"] / (1024 * 1024)  # Convert bytes → MB

# Plot CPU Usage
plt.figure(figsize=(12, 5))
plt.plot(cpu_df["timestamp"], cpu_df["cpu_usage"], label="CPU Usage", color="blue", marker='o')
plt.xlabel("Time")
plt.ylabel("CPU Usage (cores)")
plt.xticks(rotation=45)
plt.title("CPU Usage Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot Memory Usage
plt.figure(figsize=(12, 5))
plt.plot(memory_df["timestamp"], memory_df["memory_usage"], label="Memory Usage (MB)", color="red", marker='o')
plt.xlabel("Time")
plt.ylabel("Memory Usage (MB)")
plt.xticks(rotation=45)
plt.title("Memory Usage Over Time")
plt.legend()
plt.grid(True)
plt.show()

