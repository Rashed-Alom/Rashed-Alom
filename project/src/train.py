"""
Author: Rashed
File: training a CNN architucture using cifer-10 dataset
"""

import keras
import numpy as np 

# project modules
from .. import config
from . import my_model, preprocess
model = my_model.get_model()
model.summary()

#loading data
#X_train, Y_train = preprocess.load_train_data()
x_train, y_train = preprocess.load_train_data()
print("train data shape: ", x_train.shape)
print("train data label: ", y_train.shape)

#loading model


#compile
model.compile(optimizer= keras.optimizers.Adam(config.lr),
           loss= keras.losses.categorical_crossentropy,
            metrics = ['accuracy'])

#checkpoints
model_cp = my_model.save_model_checkpoint()
early_stopping = my_model.set_early_stopping()

#for training model
"""
model.fit(X_train, Y_train, 
        batch_size = config.batch_size, 
        epochs = config.nb_epochs, 
        verbose = 2,
        shuffle = True, 
        callbacks = [model_cp],
        #callbacks = [early_stopping, model_cp],  
        validation_split = 0.2)
"""
from keras.preprocessing.image import ImageDataGenerator
#data augmentation
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        horizontal_flip=True,
        validation_split=0.2,
        )
datagen.fit(x_train)

train_set = datagen.flow(x=x_train, y=y_train, batch_size=config.batch_size, subset='training', shuffle=True)
valid_set = datagen.flow(x=x_train, y=y_train, batch_size=config.batch_size, subset='validation', shuffle=True)

if 0:
        model.fit_generator(train_set,
        epochs=config.nb_epochs,
        verbose=2,
        validation_data = valid_set,
        callbacks=[early_stopping, model_cp]
        )
else:
        model.fit_generator(train_set,
        epochs=config.nb_epochs,
        verbose=2,
        validation_data = valid_set,
        callbacks=[model_cp]
        )

print("\n\n=========================================")
print("Train finish successfully.")
print("=========================================\n\n")

