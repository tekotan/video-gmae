import os
from tqdm import tqdm
import glob
import numpy as np
import joblib
import sys
from random import shuffle
import json

os.makedirs("data", exist_ok=True)

with open("/home/ubuntu/datasets/ssv2_2025-03-03_0001/labels/labels.json", "rb") as file:
    lab = json.load(file)

with open("/home/ubuntu/datasets/ssv2_2025-03-03_0001/labels/validation.json", "rb") as file:
    val = json.load(file)

with open("/home/ubuntu/datasets/ssv2_2025-03-03_0001/labels/train.json", "rb") as file:
    train = json.load(file)

save_path = "data/ssv2_train_label.npy"
labels = []
for i in range(len(train)):
    file_name = f"/home/ubuntu/datasets/ssv2_2025-03-03_0001/20bn-something-something-v2/{train[i]['id']}.webm"
    label = lab[train[i]["template"].replace("[","").replace("]","")]
    labels.append([file_name, label])
np.save(save_path, labels)

save_path = "data/ssv2_val_label.npy"
labels = []
for i in range(len(val)):
    file_name = f"/home/ubuntu/datasets/ssv2_2025-03-03_0001/20bn-something-something-v2/{val[i]['id']}.webm"
    label = lab[val[i]["template"].replace("[","").replace("]","")]
    labels.append([file_name, label])

np.save(save_path, labels)