import subprocess

#print("📦 Step 1: Fetching live metrics...")
#subprocess.run(["python3", "scripts/fetch_live_metrics.py"])

print("\n🧠 Step 2: Training model...")
subprocess.run(["python3", "scripts/train_model_live.py"])

print("\n🔮 Step 3: Making predictions...")
subprocess.run(["python3", "scripts/predictgemini.py"])

