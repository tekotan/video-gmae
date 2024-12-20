import os
from tqdm import tqdm
import glob
import numpy as np
import joblib
import sys
from random import shuffle

os.makedirs("data", exist_ok=True)
root_path = sys.argv[1] # root_path = "/datasets/kinetics_2024-01-05_0047"
train_labels = list(map(lambda x: x.split("/")[-1], glob.glob(root_path + "/train_256/*")))
val_labels = list(map(lambda x: x.split("/")[-1], glob.glob(root_path + "/val_256/*")))

save_path = "data/kinetics_train_label.npy"
labels = []
for lab in tqdm(train_labels):
    for file in glob.glob(root_path + "/train_256/" + lab + "/*.mp4"):
        labels.append([file, train_labels.index(file.split("/")[-2])])

labels = np.array(labels)
np.save(save_path, labels)


save_path = "data/kinetics_val_label.npy"
labels = []
for lab in tqdm(val_labels):
    for file in glob.glob(root_path + "/val_256/" + lab + "/*.mp4"):
        labels.append([file, val_labels.index(file.split("/")[-2])])
        
labels = np.array(labels)
np.save(save_path, labels)