import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv('data/CSVs/train.csv')
val_df = pd.read_csv('data/CSVs/val.csv')
df_test = pd.read_csv('data/CSVs/test.csv')

counts = {
    "train": len(train_df),
    "val": len(val_df),
    "test": len(df_test)
}

plt.bar(counts.keys(), counts.values())
plt.title("Sample Distribution Across Splits")
plt.ylabel("Number of Samples")
plt.savefig("bar_chart.png")
plt.show()