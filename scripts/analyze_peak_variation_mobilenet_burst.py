import pandas as pd
import glob

print("\nMobileNet - Burst Load Variation:\n")

for file in sorted(glob.glob("gpu_metrics_mobilenet_burst_run*.csv")):
    df = pd.read_csv(file)

    #  Robust column detection (handles accidental whitespace, capitalizations, etc.)
    power_col = next((col for col in df.columns if "power" in col.lower()), None)

    if power_col is None:
        print(f"{file} →  Power column not found!")
        continue

    # Clean + convert power readings
    powers = df[power_col].astype(str).str.replace(" W", "").str.strip().astype(float)

    print(f"{file} → Peak: {powers.max():.2f} W | Avg: {powers.mean():.2f} W")


