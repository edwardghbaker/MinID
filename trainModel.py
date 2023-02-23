#%% import libraries

import tensorflow as tf 
import numpy as np
import scipy as scipy 
import matplotlib as mpl 

#%% import data

data = pd.ExcelFile(r"C:\Users\User\Documents\GitHub\MinID\LEPR.xlsx")
names = data.sheet_names
test2 = pd.ExcelFile(r"C:\Users\User\Documents\GitHub\MinID\test.xlsx")

# %%
