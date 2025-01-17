import os
from tqdm import tqdm
import glob
import numpy as np
import joblib
import sys
from random import shuffle

os.makedirs("data", exist_ok=True)
root_path = sys.argv[1] # root_path = "/datasets01/Ego4D/072522/"
videos = glob.glob(root_path + "/*/*.mp4")
shuffle(videos)


print(len(videos))

save_path = "data/ego4d_train_label.npy"
labels = []
for file in tqdm(videos[:int(len(videos)*0.9)]):
    labels.append([file, -1])
labels = np.array(labels)
np.save(save_path, labels)


save_path = "data/ego4d_val_label.npy"
labels = []
for file in tqdm(videos[int(len(videos)*0.9):]):
    labels.append([file, -1])
labels = np.array(labels)
np.save(save_path, labels)