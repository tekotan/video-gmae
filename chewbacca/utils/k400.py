import os
from tqdm import tqdm
import glob
import numpy as np
import joblib
import sys

os.makedirs("data", exist_ok=True)
root_path_ = sys.argv[1] # root_path = "/datasets01/kinetics/092121/400/"

save_path = "data/kinetics_400_train_label.npy"
train = True
class_labels = {}

root_path = root_path_ + "/train/"
folders = os.listdir(root_path)
frames = []
labels = []
for folder in tqdm(folders):
    folder_path = root_path + "/" + folder
    frames_per_folder = glob.glob(folder_path + "/*.mp4")
    if(train):
        if(folder not in class_labels.keys()):
            class_labels[folder] = len(class_labels.keys())

    frames += frames_per_folder
    for fpf in frames_per_folder:
        if os.path.islink(fpf):
            fpf = os.readlink(fpf)
        labels.append([fpf, class_labels[folder]])

frames = np.array(frames)
frames = frames.reshape(frames.shape[0], 1)

labels = np.array(labels)
np.save(save_path, labels)
if(train):
    joblib.dump(class_labels, "data/class_labels_k400.pkl")



save_path = "data/kinetics_400_val_label.npy"
train = False
class_labels = joblib.load("data/class_labels_k400.pkl")

root_path = root_path_ + "/val/"
folders = os.listdir(root_path)
frames = []
labels = []
for folder in tqdm(folders):
    folder_path = root_path + "/" + folder
    frames_per_folder = glob.glob(folder_path + "/*.mp4")
    if(train):
        if(folder not in class_labels.keys()):
            class_labels[folder] = len(class_labels.keys())

    frames += frames_per_folder
    for fpf in frames_per_folder:
        if os.path.islink(fpf):
            fpf = os.readlink(fpf)
        labels.append([fpf, class_labels[folder]])

frames = np.array(frames)
frames = frames.reshape(frames.shape[0], 1)

labels = np.array(labels)
np.save(save_path, labels)
if(train):
    joblib.dump(class_labels, "data/class_labels_k400.pkl")
