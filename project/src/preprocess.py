"""
Author: Rashed
File: preprocessing cifer-10 train and test images
"""

import numpy as np
import cv2, os 
import pandas as pd 

from sklearn.preprocessing import LabelBinarizer

# project package
from .. import config 

X_train = np.ndarray((config.nb_train_samples,
                config.img_size, config.img_size,
                config.img_channel), dtype=np.float32)


def normalization(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x



#loading train data
def load_train_data():
    train_data_dir = os.path.join(config.dataset_path(), "train")
    
    train_images = sorted(os.listdir(train_data_dir),
            key = lambda x: int (x.split(".")[0]))

    train_images = [os.path.join(train_data_dir, img_path) 
                    for img_path in train_images]

    # loading images labels
    train_labels_df = pd.read_csv(os.path.join(config.dataset_path(),
                        "trainLabels.csv"))

    train_labels = train_labels_df["label"].values
    
    # one-hot encoding image labels
    encoder = LabelBinarizer()
    Y_labels = encoder.fit_transform(train_labels) 

    print("loading train images ....")
    # loading image from absolute directory using opencv
    for i, img_dir in enumerate(train_images):
        img = cv2.imread(img_dir)
        img = normalization(img)
        X_train[i] = img 

    return X_train, Y_labels


#loading test data
def load_test_data():
    test_data_dir = os.path.join(config.dataset_path(), "test")

    test_images = sorted(os.listdir(test_data_dir),
            key = lambda x: int (x.split(".")[0]))

    test_images = [os.path.join(test_data_dir, img_path) 
                    for img_path in test_images]

    #print(test_images)
    start = 0
    nb_images = 50000
    for part in range(0, 6):
        if not (part == 0): start += 50000
        end += 50000

        print(start, " ", end) 
"""
    print("loading test images ....")
    # loading image from absolute directory using opencv
    for i, img_dir in enumerate(test_images):
        img = cv2.imread(img_dir)
        img = normalization(img)
        X_test[i] = img 

    return X_train, Y_labels
    """

if __name__ == "__main__":
    load_test_data()