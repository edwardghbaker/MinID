#%% import libraries

import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split
import xgboost as xgb

import numpy as np
import scipy as scipy 
import pandas as pd

#%% import data

data = pd.ExcelFile(r"C:\Users\User\Documents\GitHub\MinID\LEPR.xlsx")
names = data.sheet_names[1:] # remove experimental sheet 
test2 = pd.ExcelFile(r"C:\Users\User\Documents\GitHub\MinID\test.xlsx")

lengths = [len(data.parse(i)) for i in names]
totalLength = np.sum(lengths)

# %% concatenate data and split into arrays

data_full = pd.concat([data.parse(sheet) for sheet in names], ignore_index=True)
data_full = np.nan_to_num(data_full)
labels = np.array([])

for i,j in zip(names,lengths):
    x = np.full(j,i)
    labels = np.append(labels,x)

#convert the strings to integers for model training

y, label = pd.factorize(labels)
#y = tf.keras.utils.to_categorical(y, num_classes=None)

'''
Now we have all the data stacked in a single array, 'data_full', and the labels in a single array, 'labels'.

'''

# %% preprocess then split data into training and testing

layer = layers.Normalization()
layer.adapt(data_full)
normalized_data = layer(data_full)

X_train, X_test, y_train, y_test = train_test_split(data_full, y, test_size=0.2)



train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# %% Make NN model


model = keras.Sequential(
    [
        layers.Dense(256, activation="relu", name="layer1"),
        layers.Dense(256, activation="relu", name="layer2"),
        layers.Dense(128, activation="relu", name="layer3"),
        layers.Dense(59)
    ]
)

batch_size = 64
epochs = 10
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    , optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    , metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#%% Make DGTree model 

#%% make naive Bayes model

#%% make KNN model 

#%% make xgb model


#%% evaluate model

test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

predictions = model.predict(X_test)

# %%
