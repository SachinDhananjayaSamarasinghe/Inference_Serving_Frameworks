import pandas as pd
import matplotlib.pyplot as plt
import glob

# Load the steadyload CSVs
csv_files = sorted(glob.glob("gpu_metrics_mobilenet_steady_run*.csv"))
labels = [f"Run {i+1}" for i in range(len(csv_files))]

peak_powers = []
avg_powers = []

for file in csv_files:
    df = pd.read_csv(file)
    col = [c for c in df.columns if "power.draw" in c.lower()][0]
    df[col] = df[col].astype(str).str.replace(" W", "").str.strip().astype(float)

    peak_powers.append(df[col].max())
    avg_powers.append(df[col].mean())

# Plot
x = range(len(csv_files))
plt.figure(figsize=(10, 6))
plt.plot(x, peak_powers, label="Peak Power (W)", marker='o')
plt.plot(x, avg_powers, label="Average Power (W)", marker='x')
plt.xticks(x, labels)
plt.title("MobileNet Steady Load - Power Variation Across 3 Runs")
plt.ylabel("Power (Watts)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mobilenet_steadyload_variation.png")
plt.show()


