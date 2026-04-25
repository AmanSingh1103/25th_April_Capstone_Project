import pandas as pd
import numpy as np

def detect_drift(old_data, new_data, threshold=0.25):

    old_mean = np.mean(old_data)
    new_mean = np.mean(new_data)

    if old_mean == 0:
        return False

    drift_score = abs(new_mean - old_mean) / old_mean

    print("📊 Drift Score:", drift_score)

    if drift_score > threshold:
        print("⚠️ DATA DRIFT DETECTED!")
        return True
    else:
        print("✅ No Drift")
        return False


if __name__ == "__main__":

    df = pd.read_csv("data/retail.csv")

    old = df["UnitsSold"][:5]
    new = df["UnitsSold"][5:]

    detect_drift(old, new)
    