#%% import libraries

import tensorflow as tf 
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

import numpy as np
import scipy as scipy 
import pandas as pd
from joblib import dump, load

#%% import data

data = pd.ExcelFile("TrainingData\\LEPR.xlsx")
names = data.sheet_names[1:] # remove experimental sheet 
test2 = pd.ExcelFile("TrainingData\\test.xlsx")

lengths = [len(data.parse(i).drop(columns=['Index']).dropna(axis = 0, how = 'all')) for i in names]
for i,j in zip(names,lengths):
    if j < 100:
        names.remove(i)
        lengths.remove(j)

totalLength = np.sum(lengths)
# print(totalLength)
# print(len(names))
# %% concatenate data and split into arrays

data_full = pd.concat([data.parse(i).drop(columns=['Index']).dropna(axis = 0, how = 'all') for i in names], ignore_index=True)
data_full = np.nan_to_num(data_full)
labels = np.array([])

for i,j in zip(names,lengths):
    x = np.full(j,i)
    labels = np.append(labels,x)

#convert the strings to integers for model training

y, label = pd.factorize(labels)
#y = tf.keras.utils.to_categorical(y, num_classes=None)

'''Now we have all the data stacked in a single array, 'data_full', and the labels in a single array, 'labels'.'''

# %% preprocess then split data into training and testing

layer = layers.Normalization()
layer.adapt(data_full)
normalized_data = layer(data_full)

X_train, X_test, y_train, y_test = train_test_split(data_full, y, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# %% Make NN model

nn_model = keras.Sequential(
    [
        layers.Dense(256, activation="relu", name="layer1"),
        layers.Dense(256, activation="relu", name="layer2"),
        layers.Dense(128, activation="relu", name="layer3"),
        layers.Dense(len(names), activation="softmax", name="Output")
    ]
)

batch_size = 64
epochs = 10
nn_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    , optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
    , metrics=["accuracy"])

nn_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

nn_model.save("nn_model.h5")
#%% make xgb model

xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(names))
xgb_model.fit(X_train, y_train)
xgb_model.save_model("xgb_model.json")
#%% make naive Bayes model

cnb_model = CategoricalNB()
cnb_model.fit(X_train, y_train)
dump(cnb_model, 'cnb_model.joblib')

#%% make KNN model 

knn_model = KNeighborsClassifier(n_neighbors=len(names))
knn_model.fit(X_train,y_train)
dump(knn_model, 'knn_model.joblib')
