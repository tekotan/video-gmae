import os
from tqdm import tqdm
import glob
import numpy as np
import joblib
import sys

os.makedirs("data", exist_ok=True)
root_path = sys.argv[1] # /datasets01/imagenet_full_size/061417/train

save_path = "data/imagenet_train_label.npy"
train = True
class_labels = {}

folders = os.listdir(root_path)
frames = []
labels = []
for folder in tqdm(folders):
    folder_path = root_path + "/" + folder
    frames_per_folder = glob.glob(folder_path + "/*.JPEG")
    if(train):
        if(folder not in class_labels.keys()):
            class_labels[folder] = len(class_labels.keys())

    frames += frames_per_folder
    for fpf in frames_per_folder:
        labels.append([fpf, class_labels[folder]])

frames = np.array(frames)
frames = frames.reshape(frames.shape[0], 1)

labels = np.array(labels)
np.save(save_path, labels)
if(train):
    joblib.dump(class_labels, "data/class_labels_in1k.pkl")




save_path = "data/imagenet_val_label.npy"
train = False
class_labels = joblib.load("data/class_labels_in1k.pkl")

folders = os.listdir(root_path)
frames = []
labels = []
for folder in tqdm(folders):
    folder_path = root_path + "/" + folder
    frames_per_folder = glob.glob(folder_path + "/*.JPEG")
    if(train):
        if(folder not in class_labels.keys()):
            class_labels[folder] = len(class_labels.keys())

    frames += frames_per_folder
    for fpf in frames_per_folder:
        labels.append([fpf, class_labels[folder]])

frames = np.array(frames)
frames = frames.reshape(frames.shape[0], 1)

labels = np.array(labels)
np.save(save_path, labels)
if(train):
    joblib.dump(class_labels, "data/class_labels_in1k.pkl")

