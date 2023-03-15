#%% import libraries

import tensorflow as tf 
from tf.data import Dataset
import numpy as np
import scipy as scipy 
import matplotlib as mpl 
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


# %% split data into training and testing

Dataset(data_full,labels)