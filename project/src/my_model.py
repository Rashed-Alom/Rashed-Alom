"""
Author: Rashed
File: Network architecture for classifing cifer-10 data images
"""

import keras, os
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.models import Sequential, load_model

 # project modules
from .. import config

 #defining CNN model
def get_model():
     model = Sequential()
     model.add(Conv2D(32, (3, 3), padding = "same",
                input_shape = config.img_shape))

    model.add(Activation("relu"))

    return model





    if __name__ == "__main__":
        m = get_model()
        m.summary()
