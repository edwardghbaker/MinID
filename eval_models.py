#%% Import libraries
import tensorflow as tf
import xgboost as xgb
from tensorflow import keras

import numpy as np
import scipy as scipy 
import pandas as pd
from joblib import dump, load

#%% Load in models
xgb_model = xgb.XGBClassifier().load_model('Models\\xgb_model.json')
knn_model = load('Models\\knn_model.joblib')
cnb_model = load('Models\\cnb_model.joblib')

nn_model = tf.keras.models.load_model("Models\\nn_model.h5")

#%% Load in data



#%% Evaluate models
test_scores = nn_model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

# %%

predictions = nn_model.predict(X_test)
