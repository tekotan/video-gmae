import os
from tqdm import tqdm
import glob
import numpy as np
import joblib
import sys

os.makedirs("data", exist_ok=True)
root_path = sys.argv[1] # root_path = "/datasets01/kinetics/092121/400/"
videos = glob.glob(root_path + "/*.avi")
print(len(videos))
save_path = "data/cater_train_label.npy"
labels = []
for file in tqdm(videos[:int(len(videos)*0.8)]):
    labels.append([file, 0])
labels = np.array(labels)
np.save(save_path, labels)


save_path = "data/cater_val_label.npy"
labels = []
for file in tqdm(videos[int(len(videos)*0.8):]):
    labels.append([file, 0])
labels = np.array(labels)
np.save(save_path, labels)