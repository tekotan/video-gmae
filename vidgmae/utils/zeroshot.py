import os
from tqdm import tqdm
import glob
import numpy as np
import joblib
import sys
from random import shuffle

os.makedirs("data", exist_ok=True)
root_path = "/home/tekotan/Vidgmae_test/data-download/zeroshot-midtraining/kubric"
videos = glob.glob(root_path + "/train/*.mp4")
print(len(videos))
save_path = "data/kubric_train_label.npy"
labels = []
for file in tqdm(videos):
    labels.append([file, -1])
labels = np.array(labels)
np.save(save_path, labels)


videos = glob.glob(root_path + "/validation/*.mp4")
print(len(videos))
save_path = "data/kubric_val_label.npy"
labels = []
for file in tqdm(videos):
    labels.append([file, -1])
labels = np.array(labels)
np.save(save_path, labels)

root_path = "/home/tekotan/Vidgmae_test/data-download/zeroshot-midtraining/davis"
videos = glob.glob(root_path + "/*.mp4")
print(len(videos))
save_path = "data/davis_label.npy"
labels = []
for file in tqdm(videos):
    labels.append([file, -1])
labels = np.array(labels)
np.save(save_path, labels)
