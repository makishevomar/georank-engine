import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(reference_data_path="data/raw_dataset.csv", num_new_requests=100, drift_scenario=False):
    # 1. Load baseline distribution
    ref_df = pd.read_csv(reference_data_path)
    ref_distances = ref_df["distance_meters"].values
    
    # 2. Simulate incoming production traffic
    if drift_scenario:
        # Simulate drift: e.g. bad weather causes users to select points much further away
        print("Simulating bad weather (generating abnormally far pickup points)...")
        current_distances = np.random.uniform(300.0, 1500.0, size=num_new_requests)
    else:
        # Normal traffic matching the training distribution
        print("Simulating normal day (distribution is stable)...")
        current_distances = np.random.uniform(0.0, 500.0, size=num_new_requests)
        
    # 3. Apply Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(ref_distances, current_distances)
    
    print("\n=== MLOps Drift Monitoring Results ===")
    print(f"KS Statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.8f}")
    
    if p_value < 0.05:
        print("ALERT: Data Drift detected on feature [distance_meters]!")
        print("Action required: Trigger automated model retraining pipeline.")
    else:
        print("Feature distribution is stable. Model operates in normal bounds.")

if __name__ == "__main__":
    print("--- TEST 1 (Normal Traffic) ---")
    detect_drift(drift_scenario=False)
    
    print("\n" + "="*40 + "\n")
    
    print("--- TEST 2 (Drifted Traffic) ---")
    detect_drift(drift_scenario=True)
