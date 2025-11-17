import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import os
import pandas as pd


main_dir = "./data"
xrays_dir = os.path.join(main_dir, "xrays")
masks_dir = os.path.join(main_dir, "masks")


xray_files = sorted([f for f in os.listdir(xrays_dir) if f.endswith(".png")])
mask_files = set([f for f in os.listdir(masks_dir) if f.endswith(".png")])


metadata = pd.DataFrame(columns=["xrays", "masks"])

for xray in xray_files:
    xray_path = os.path.join(xrays_dir, xray)
    mask_path = os.path.join(masks_dir, xray) if xray in mask_files else None
    metadata.loc[len(metadata)] = [xray_path, mask_path]


csv_dir = os.path.join(main_dir, "CSVs")
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, "dataset.csv")
metadata.to_csv(csv_path, index=False)

print(f"dataset.csv created successfully with {len(metadata)} rows.")
print(f"Saved at: {csv_path}")
print(f"Found {metadata['masks'].isna().sum()} X-rays without masks.")


