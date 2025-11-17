import pandas as pd
from sklearn.model_selection import train_test_split
import os

# === Load the main dataset ===
dataset_path = "data/CSVs/dataset.csv"
df = pd.read_csv(dataset_path)

# === Identify test and labeled samples ===
df_test = df[df['masks'].isna()]  # 50 unlabeled samples (mask=None)
df_labeled = df[df['masks'].notna()]  # 100 labeled samples

# === Split labeled data into train and val ===
train_df, val_df = train_test_split(df_labeled, test_size=0.2, random_state=42)

# === Save all CSVs ===
output_dir = "data/CSVs"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print("Files saved:")
print(f"- train.csv ({len(train_df)} samples)")
print(f"- val.csv ({len(val_df)} samples)")
print(f"- test.csv ({len(df_test)} samples)")