"""
Author: Rashed
File: testing & submitting kaggle cifer-10 test images
"""

import numpy as np 
import pandas as pd 
import os

# project modules 
from .. import config
from . import preprocess, my_model

# loading model
model = my_model.read_model()

# loading test data
result = []
for part in range (0, 1)
    X_test = preprocess.get_test_data_by_part(part)

    #predicting results
    print("predecting results")
    predictions = model.predict(X_test,
                                batch_size = config.batch_size,
                                verbose = 2)

    print(len(predictions))