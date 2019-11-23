"""
Author: Rashed
File: Network architecture for classifing cifer-10 data images
"""

import keras, os
from keras.layers import (Dense, Activation, 
                    Flatten, Conv2D, MaxPooling2D, Dropout)
from keras.models import Sequential, load_model

# project modules
from .. import config


#defining CNN model
def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding = "same",
                input_shape = config.img_shape))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3), padding = "same",
                input_shape = config.img_shape))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(rate = 0.20))

    model.add(Conv2D(64, (3, 3), padding = "same",
                input_shape = config.img_shape))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3), padding = "same",
                input_shape = config.img_shape))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(rate = 0.20))

    model.add(Flatten())
    model.add(Dense(384, 
                kernel_regularizer= keras.regularizers.l2(0.01)))
    model.add(Activation("relu"))

    model.add(Dropout(rate = 0.30))
    model.add(Dense(config.nb_classes))
    model.add(Activation("softmax"))

    return model




if __name__ == "__main__":
    m = get_model()
    m.summary()
