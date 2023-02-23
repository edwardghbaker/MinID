#%% import libraries

import tensorflow as tf 
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
labels = np.zeros((totalLength, 1))

for i in range(len(names)):
    labels[i*len(data.parse(names[i])):(i+1)*len(data.parse(names[i]))] = names[i]