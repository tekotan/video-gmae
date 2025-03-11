import os
from tqdm import tqdm
import glob
import numpy as np
import joblib
import sys
from random import shuffle

os.makedirs("data", exist_ok=True)
root_path = sys.argv[1] # root_path = "/datasets01/kinetics/092121/400/"
videos = glob.glob(root_path + "/*/*.avi")
print(len(videos))

all_labels = list(set(map(lambda x: x.split("/")[-1].split("_")[1], videos)))
print(len(videos))

save_path = "data/ucf101_train_label.npy"
labels = []
for file in tqdm(videos[:int(len(videos)*0.8)]):
    labels.append([file, all_labels.index(file.split("/")[-1].split("_")[1])])
labels = np.array(labels)
np.save(save_path, labels)


save_path = "data/ucf101_val_label.npy"
labels = []
for file in tqdm(videos[int(len(videos)*0.8):]):
    labels.append([file, all_labels.index(file.split("/")[-1].split("_")[1])])
labels = np.array(labels)
np.save(save_path, labels)