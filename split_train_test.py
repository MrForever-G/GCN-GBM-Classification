import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


sample_info = pd.read_csv("./GBM/final_sample_id.csv", index_col=None)
# sample_info = sample_info[sample_info["GeneExp_Subtype"].isin(["Mesenchymal", "Proneural"])].copy() # keep only two classes

sample_num = sample_info.shape[0]
labels = sample_info["GeneExp_Subtype"]
label_type = {'Classical': 0, 'Mesenchymal': 1, 'Neural': 2, 'Proneural': 3}

label_vals = np.array([label_type[l] for l in labels])

# labels = sample_info["GeneExp_Subtype"]
# bin_map = {"Proneural": 0, "Mesenchymal": 1}
# label_vals = labels.map(bin_map).to_numpy()

train_val_idx, test_idx = train_test_split(np.arange(sample_num), test_size=0.3, random_state=2024, stratify=label_vals)

train_sample = sample_info.iloc[train_val_idx]
test_sample = sample_info.iloc[test_idx]

train_sample.to_csv("./GBM/train_sample_id.csv", index=None)
test_sample.to_csv("./GBM/test_sample_id.csv", index=None)


train_labels_count = train_sample["GeneExp_Subtype"].value_counts()
test_labels_count = test_sample["GeneExp_Subtype"].value_counts()
print("Train label distribution:")
print(train_labels_count)
print(f"Total train samples: {train_sample.shape[0]}")
print("--------------------------------")
print("Test label distribution:")
print(test_labels_count)
print(f"Total test samples: {test_sample.shape[0]}")

