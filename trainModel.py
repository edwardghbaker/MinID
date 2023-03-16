#%% import libraries

import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split

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
labels = np.array([])

for i,j in zip(names,lengths):
    x = np.full(j,i)
    labels = np.append(labels,x)

#convert the strings to integers for model training

y, label = pd.factorize(labels)

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

# %% create model


model = keras.Sequential(
    [
        layers.Dense(256, activation="relu", name="layer1"),
        layers.Dense(256, activation="relu", name="layer2"),
        layers.Dense(128, activation="relu", name="layer3"),
        tf.keras.layers.Dense(59, activation='sigmoid')
    ]
)


# %% compile model
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

#%% evaluate model
