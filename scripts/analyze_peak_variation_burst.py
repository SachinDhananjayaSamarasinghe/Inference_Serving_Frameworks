import pandas as pd
import glob

csv_files = sorted(glob.glob("gpu_metrics_resnet_burst_run*.csv"))

print("\n Multi-Run Peak Power Comparison (ResNet50 - Low Load):\n")
for file in csv_files:
    df = pd.read_csv(file)

    # Strip whitespace and ' W' suffix from column name (and values)
    col = [c for c in df.columns if "power.draw" in c.lower()][0]
    df[col] = df[col].astype(str).str.replace(" W", "").str.strip()

    powers = df[col].dropna().astype(float)

    peak = powers.max()
    avg = powers.mean()
    print(f"{file} → Peak: {peak:.2f} W | Avg: {avg:.2f} W")

