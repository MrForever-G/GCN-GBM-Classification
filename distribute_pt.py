import os
import pandas as pd
import shutil
from utils import check_dir

train_sample_info = pd.read_csv("./GBM/train_sample_id.csv", index_col=None)
test_sample_info = pd.read_csv("./GBM/test_sample_id.csv", index_col=None)


train_sample = train_sample_info["sampleID"].tolist()
test_sample = test_sample_info["sampleID"].tolist()


source_path = "./GBM/postdata/graph_structure/"
dest_path = "./GBM/postdata/train_test_split/"

check_dir(dest_path)
check_dir(os.path.join(dest_path, "train"))
check_dir(os.path.join(dest_path, "test"))

all_sample = [f for f in os.listdir(source_path) 
              if f.endswith(".pt")]

for sample in all_sample:
    sample_id = sample.split("_")[0]
    if sample_id in train_sample:
        shutil.move(os.path.join(source_path, sample), 
                    os.path.join(dest_path, "train", sample))
    elif sample_id in test_sample:
        shutil.move(os.path.join(source_path, sample), 
                    os.path.join(dest_path, "test", sample))
    else:
        print(f"Sample {sample_id} not found in train or test lists.")


