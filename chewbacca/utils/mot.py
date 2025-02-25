import os
from tqdm import tqdm
import glob
import numpy as np
import joblib
import sys
from random import shuffle

os.makedirs("data", exist_ok=True)
root_path = sys.argv[1] # root_path = "/datasets/motsynth_2024-01-10_1813/"
videos = glob.glob(root_path + "MOTSynth_[0-9]/*.mp4")
shuffle(videos)


print(len(videos))

save_path = "data/mot_train_label.npy"
labels = []
for file in tqdm(videos[:int(len(videos)*0.8)]):
    labels.append([file, -1])
labels = np.array(labels)
np.save(save_path, labels)


save_path = "data/mot_val_label.npy"
labels = []
for file in tqdm(videos[int(len(videos)*0.8):]):
    labels.append([file, -1])
labels = np.array(labels)
np.save(save_path, labels)