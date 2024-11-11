import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from keras.utils import to_categorical
from keras.layers import (
    Dense,
    Input,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Activation,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adamax
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras import backend as k

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

raw_folder = "data/raw/Bangla Alphabets"
train_dir = ""
valid_dir = ""
test_dir = ""

IMG_SIZE = 32
BATCH_SIZE = 32
LOSS_FN = "CrossEntropyLoss"
OPTIMIZER_LIST = ["adam", "RMSprop"]
DROPOUT_LIST = [0.25, 0.5]
LEAENING_RATE_LIST = [0.001, 0.0001]
EPOCHS = 10

NUM_CLASSES = 50

# Declaring constants
FIG_WIDTH = 20  # Width of figure
HEIGHT_PER_ROW = (
    3  # Height of each row when showing a figure which consists of multiple rows
)
RESIZE_DIM = 32  # The images will be resized to 28x28 pixels


def get_key(path):
    # seperates the key of an image from the filepath
    key = path.split(sep=os.sep)[-1]
    return key


def get_data(paths_img, path_label=None, resize_dim=None):
    """reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    """
    X = []  # initialize empty list for resized images
    for i, path in enumerate(paths_img):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # images loaded in color (BGR)
        # ret,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # img = cv2.bilateralFilter(img,9,75,75)
        # img = cv2.medianBlur(img,5)
        # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # cnahging colorspace to GRAY
        if resize_dim is not None:
            img = cv2.resize(
                img, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA
            )  # resize image to 28x28
        # X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)  # unblur
        img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # filter
        img = cv2.filter2D(img, -1, kernel)
        # thresh = 200
        # maxValue = 255
        # th, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
        # ret,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        X.append(img)  # expand image to 28x28x1 and append to the list
        # display progress
        if i == len(paths_img) - 1:
            end = "\n"
        else:
            end = "\r"
        print("processed {}/{}".format(i + 1, len(paths_img)), end=end)

    X = np.array(X)  # tranform list to numpy array
    if path_label is None:
        return X
    else:
        df = pd.read_csv(path_label)  # read labels
        df = df.set_index("filename")
        y_label = [
            df.loc[get_key(path)]["digit"] for path in paths_img
        ]  # get the labels corresponding to the images
        y = to_categorical(
            y_label, 10
        )  # transfrom integer value to categorical variable
        return X, y


def imshow_group(X, y, y_pred=None, n_per_row=10, phase="processed"):
    """helper function to visualize a group of images along with their categorical true labels (y) and prediction probabilities.
    Args:
        X: images
        y: categorical true labels
        y_pred: predicted class probabilities
        n_per_row: number of images per row to be plotted
        phase: If the images are plotted after resizing, pass 'processed' to phase argument.
            It will plot the image and its true label. If the image is plotted after prediction
            phase, pass predicted class probabilities to y_pred and 'prediction' to the phase argument.
            It will plot the image, the true label, and it's top 3 predictions with highest probabilities.
    """
    n_sample = len(X)
    img_dim = X.shape[1]
    j = np.ceil(n_sample / n_per_row)
    fig = plt.figure(figsize=(FIG_WIDTH, HEIGHT_PER_ROW * j))
    for i, img in enumerate(X):
        plt.subplot(j, n_per_row, i + 1)
        #         img_sq=np.squeeze(img,axis=2)
        #         plt.imshow(img_sq,cmap='gray')
        plt.imshow(img)
        if phase == "processed":
            plt.title(np.argmax(y[i]))
        if phase == "prediction":
            top_n = 3  # top 3 predictions with highest probabilities
            ind_sorted = np.argsort(y_pred[i])[::-1]
            h = img_dim + 4
            for k in range(top_n):
                string = "pred: {} ({:.0f}%)\n".format(
                    ind_sorted[k], y_pred[i, ind_sorted[k]] * 100
                )
                plt.text(
                    img_dim / 2,
                    h,
                    string,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                h += 4
            if y is not None:
                plt.text(
                    img_dim / 2,
                    -4,
                    "true label: {}".format(np.argmax(y[i])),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
        plt.axis("off")
    plt.show()


def create_submission(predictions, keys, path):
    result = pd.DataFrame(predictions, columns=["label"], index=keys)
    result.index.name = "key"
    result.to_csv(path, index=True)
