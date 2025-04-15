import subprocess
import time
import os
import signal

# Start fetching metrics as a background process
print("📦 Starting live metrics fetcher in background...")
fetch_process = subprocess.Popen(["python3", "scripts/fetch_live_metrics.py"])

# Wait 10 seconds to allow some data to be collected (you can increase this if needed)
print("⏳ Waiting for metrics to be fetched...")
time.sleep(10)

# Train the model
print("🧠 Training model...")
subprocess.run(["python3", "scripts/train_model_live.py"])

# Predict
print("🔮 Making predictions...")
subprocess.run(["python3", "scripts/predictgemini.py"])

# Optionally stop the fetch process (if you only want a demo)
print("🛑 Stopping metrics fetcher process...")
os.kill(fetch_process.pid, signal.SIGTERM)

print("✅ Pipeline demo complete!")

