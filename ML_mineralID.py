# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:04:28 2021

@author: user
"""
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import time
import sys
import math as m
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scipy.optimize as sciop
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.pyplot import figure
import matplotlib as mpl
import mpl_toolkits as mpltk
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib import transforms
from tqdm import tqdm

#%% Load the data and delete all the nan variables 

class MinID():
    
    # def __init__(self,trainingData,unknownData):
    #     self.trainingData = trainingData
    #     self.unknownData = unknownData
#        return trainingData, unknownData
    
    def reorderAndRelabel(self,unknownData):
        try:
            unknownData = pd.read_csv(unknownData)
        except: 
            unknownData = pd.read_excel(unknownData)
        # except:
        #     print('Bad input format')
        #     return None
        
        cols = unknownData.columns
        eles = ['Wt: SiO2','Wt: TiO2','Wt: Al2O3','Wt: Fe2O3','Wt: Cr2O3','Wt: FeO','Wt: MnO','Wt: MgO','Wt: NiO',
       'Wt: CaO','Wt: Na2O','Wt: K2O','Wt: P2O5','Wt: H2O']
        oxides = ['SiO2','TiO2','Al2O3','Fe2O3','Cr2O3','FeO','MnO','MgO','NiO',
       'CaO','Na2O','K2O','P2O5','H2O']
        
        for i in range(len(eles)):
            oxide = oxides[i]
            ele = eles[i]
            newname = [string for string in cols if oxide.casefold() in string.casefold()]
            unknownDataSorted = unknownData.rename({str(newname) : ele},axis='columns')
            
        return unknownDataSorted            
        

    def train(self,trainingData):
        xls = pd.ExcelFile(trainingData)
        sheets = xls.sheet_names
        names = []
        
        # Here i select the columns that i want to fit the model on
        ele = ['Wt: SiO2','Wt: TiO2','Wt: Al2O3','Wt: Fe2O3','Wt: Cr2O3','Wt: FeO','Wt: MnO','Wt: MgO','Wt: NiO',
       'Wt: CaO','Wt: Na2O','Wt: K2O','Wt: P2O5','Wt: H2O']
        
        X = pd.DataFrame([])
        # creating a blank dataframe to assign data to 
        for i in sheets:
            j = i.replace(' ','')
            k = j.replace('-','')
            names.append(k)
        # making a list of sheet names for the labels 
        for i in tqdm(range((len(sheets)-1))):
            data = pd.read_excel('LEPR.xlsx',sheet_name=sheets[i+1])
            b = pd.DataFrame(np.full(len(data), names[i+1]), columns=['label'])
            c = pd.concat([b,data[ele]],axis=1)
            X = pd.concat([X,c],axis=0)
            Y = X.dropna(subset = ["Wt: SiO2"])
            Y = Y.fillna(0)
            x = np.array(Y.iloc[:,1:])
            y = np.array(Y.iloc[:,0])    
        
        n = len(names)
        minID_KNN = KNeighborsClassifier(n_neighbors=n).fit(x,y)
        minID_NN = MLPClassifier().fit(x,y)
        
        return minID_KNN, minID_NN, names
    
    def siliconOxideRatio(self,unknownDataSorted):
        uData = unknownDataSorted

        SiORatio = pd.DataFrame([])
            
        uData["sum"] = uData.sum(axis=1)
        atomData = pd.DataFrame(index=range(len(uData)),columns=['Si','Ti','Al','Cr','Fe','Mn','Mg','Ni',
           'Ca','Na','K','P','H','O'])
        
        for i in range(len(uData)):
            
            atomData['Si'][i] = (uData['Wt: SiO2'][i]/60.08)
            atomData['Ti'][i] = (uData['Wt: TiO2'][i]/79.87)
            atomData['Al'][i] = 2*(uData['Wt: Al2O3'][i]/101.96)
            atomData['Cr'][i]= 2*(uData['Wt: Cr2O3'][i]/151.99)
            atomData['Fe'][i] = 2*(uData['Wt: Fe2O3'][i]/159.69) + (uData['Wt: FeO'][0]/71.84)
            atomData['Mn'][i] = (uData['Wt: MnO'][i]/70.94)
            atomData['Mg'][i] = (uData['Wt: MgO'][i]/40.30)
            atomData['Ni'][i] = (uData['Wt: NiO'][i]/74.69)
            atomData['Ca'][i] = (uData['Wt: CaO'][i]/56.08)
            atomData['Na'][i] = 2*(uData['Wt: Na2O'][i]/61.98)
            atomData['K'][i] = 2*(uData['Wt: K2O'][i]/94.20)
            atomData['P'][i] = 2*(uData['Wt: P2O5'][i]/141.94)
            atomData['H'][i] = 2*(uData['Wt: H2O'][i]/18.02)
            atomData['O'][i] = 2*(uData['Wt: SiO2'][i]/60.08)+2*(uData['Wt: TiO2'][i]/79.87)+3*(uData['Wt: Al2O3'][i]/101.96)+3*(uData['Wt: Cr2O3'][i]/151.99)+3*(uData['Wt: Fe2O3'][i]/159.69) + (uData['Wt: FeO'][i]/71.84)+(uData['Wt: MnO'][i]/70.94)+(uData['Wt: MgO'][i]/40.30)+(uData['Wt: NiO'][i]/74.69)+(uData['Wt: CaO'][i]/56.08)+(uData['Wt: Na2O'][i]/61.98)+(uData['Wt: K2O'][i]/94.20)+5*(uData['Wt: P2O5'][i]/141.94)+(uData['Wt: H2O'][i]/18.02)
            
            atomData = 100*atomData.div(atomData.sum(axis=1), axis=0)
   
        
        for i in range(len(uData)):
            if atomData['Si'][i]/atomData['O'][i] in range(0.23,0.27):
                SiORatio['Si/O'][i] = atomData['Si'][i]/atomData['O'][i]
                SiORatio['Classification'][i] = 'Nesosilicate'
                
            if atomData['Si'][i]/atomData['O'][i] in range(0.27,0.3):
                SiORatio['Si/O'][i] = atomData['Si'][i]/atomData['O'][i]
                SiORatio['Classification'][i] = 'Sorosilicate'
                
            if atomData['Si'][i]/atomData['O'][i] in range(0.31,0.35):
                SiORatio['Si/O'][i] = atomData['Si'][i]/atomData['O'][i]
                SiORatio['Classification'][i] = 'Cyclosilicate/Inosilicate'
                
            if atomData['Si'][i]/atomData['O'][i] in range(0.35,0.375):
                SiORatio['Si/O'][i] = atomData['Si'][i]/atomData['O'][i]
                SiORatio['Classification'][i] = 'Double-chain Inosilicate'
                
            if atomData['Si'][i]/atomData['O'][i] in range(0.375,0.425):
                SiORatio['Si/O'][i] = atomData['Si'][i]/atomData['O'][i]
                SiORatio['Classification'][i] = 'Phylosilicate'
                
            if atomData['Si'][i]/atomData['O'][i] in range(0.475,0.525):
                SiORatio['Si/O'][i] = atomData['Si'][i]/atomData['O'][i]
                SiORatio['Classification'][i] = 'Tectosilicates'
            
            else:
                SiORatio['Si/O'][i] = atomData['Si'][i]/atomData['O'][i]
                SiORatio['Classification'][i] = 'Unknown'
        return SiORatio
    
    def DHZClassification(self,unknownDataSorted):

        uData = unknownDataSorted
        DHZclass = pd.DataFrame(index=range(len(uData)),columns=['DHZ'])
            
        atomData = pd.DataFrame(index=range(len(uData)),columns=['Si','Ti','Al','Cr','Fe','Mn','Mg','Ni',
           'Ca','Na','K','P','H','O'])
            
        for i in range(len(uData)):
            atomData['Si'][i] = (uData['Wt: SiO2'][i]/60.08)
            atomData['Ti'][i] = (uData['Wt: TiO2'][i]/79.87)
            atomData['Al'][i] = 2*(uData['Wt: Al2O3'][i]/101.96)
            atomData['Cr'][i]= 2*(uData['Wt: Cr2O3'][i]/151.99)
            atomData['Fe'][i] = 2*(uData['Wt: Fe2O3'][i]/159.69) + (uData['Wt: FeO'][0]/71.84)
            atomData['Mn'][i] = (uData['Wt: MnO'][i]/70.94)
            atomData['Mg'][i] = (uData['Wt: MgO'][i]/40.30)
            atomData['Ni'][i] = (uData['Wt: NiO'][i]/74.69)
            atomData['Ca'][i] = (uData['Wt: CaO'][i]/56.08)
            atomData['Na'][i] = 2*(uData['Wt: Na2O'][i]/61.98)
            atomData['K'][i] = 2*(uData['Wt: K2O'][i]/94.20)
            atomData['P'][i] = 2*(uData['Wt: P2O5'][i]/141.94)
            atomData['H'][i] = 2*(uData['Wt: H2O'][i]/18.02)
            atomData['O'][i] = 2*(uData['Wt: SiO2'][i]/60.08)+2*(uData['Wt: TiO2'][i]/79.87)+3*(uData['Wt: Al2O3'][i]/101.96)+3*(uData['Wt: Cr2O3'][i]/151.99)+3*(uData['Wt: Fe2O3'][i]/159.69) + (uData['Wt: FeO'][i]/71.84)+(uData['Wt: MnO'][i]/70.94)+(uData['Wt: MgO'][i]/40.30)+(uData['Wt: NiO'][i]/74.69)+(uData['Wt: CaO'][i]/56.08)+(uData['Wt: Na2O'][i]/61.98)+(uData['Wt: K2O'][i]/94.20)+5*(uData['Wt: P2O5'][i]/141.94)+(uData['Wt: H2O'][i]/18.02)
        
        atomData = 100*atomData.div(atomData.sum(axis=1), axis=0)            
        
        for i in range(len(uData)):
            if 7*atomData['Si'][i] < 1.25 and 7*atomData['Si'][i] >0.975 and 7*atomData['Al'][i] < 0.5 and 7*atomData['Ti'][i] < 0.5 and 7*atomData['Ca'][i] < 0.5 and 7*(atomData['Mg'][i] + atomData['Fe'][i] + atomData['Mn'][i]) > 1.9 and 7*(atomData['Mg'][i] + atomData['Fe'][i] + atomData['Mn'][i]) < 2.1: DHZclass['DHZ'][i] = 'Olivine'
            
            if 40*(atomData['Si'][i] + atomData['Al'][i] + atomData['Ti'][i] + atomData['Cr'][i]) < 10.12 and 40*(atomData['Si'][i] + atomData['Al'][i] + atomData['Ti'][i] + atomData['Cr'][i]) > 7.76 and 40*(atomData['Mg'][i] + atomData['Fe'][i] + atomData['Mn'][i] + atomData['Ca'][i]) > 5.85 and 40*(atomData['Si'][i] + atomData['Al'][i] + atomData['Ti'][i] + atomData['Cr'][i]) < 6.45: DHZclass['DHZ'][i] = 'Garnet'
            
            if 10*(atomData['Si'][i]+atomData['Al'][i]+atomData['Ti'][i]) > 1.95 and 10*(atomData['Si'][i]+atomData['Al'][i]+atomData['Ti'][i]) < 2.05 and 10*(atomData['Fe'][i]+atomData['Cr'][i]+atomData['Mg'][i]+atomData['Ni'][i]+atomData['Mn'][i]+atomData['Ca'][i]) > 1.65 and 10*(atomData['Fe'][i]+atomData['Cr'][i]+atomData['Mg'][i]+atomData['Ni'][i]+atomData['Mn'][i]+atomData['Ca'][i]) < 2.01 and 10*atomData['Ca'][i] > 0.5: DHZclass['DHZ'][i] = 'Clinopyroxene'
            
            if 10*(atomData['Si'][i]+atomData['Al'][i]+atomData['Ti'][i]) > 1.95 and 10*(atomData['Si'][i]+atomData['Al'][i]+atomData['Ti'][i]) < 2.05 and 10*(atomData['Fe'][i]+atomData['Cr'][i]+atomData['Mg'][i]+atomData['Ni'][i]+atomData['Mn'][i]+atomData['Ca'][i]) > 1.65 and 10*(atomData['Fe'][i]+atomData['Cr'][i]+atomData['Mg'][i]+atomData['Ni'][i]+atomData['Mn'][i]+atomData['Ca'][i]) < 2.01 and 10*atomData['Ca'][i] < 0.5: DHZclass['DHZ'][i] = 'Orthopyroxene'
            
            else: DHZclass['DHZ'][i] = 'Unknown/Glass'
            
             
        return DHZclass
    
    
    def classify(self,unknownData,trainingdata):
        
        unknownDataSorted = MinID().reorderAndRelabel(unknownData)
        
        output = pd.DataFrame([])
        
        DHZclass = MinID().DHZClassification(unknownDataSorted)
        output = output.append(DHZclass)

        SiORatio = MinID().siliconOxideRatio(unknownDataSorted)
        output = output.append(SiORatio)

        minID_KNN, minID_NN, names = MinID().train(trainingdata)

        KNN = minID_KNN.predict(unknownDataSorted)
        output = output.append(KNN)

        NN = minID_NN.predict(unknownDataSorted)
        output = output.append(NN)
        
        return output
    
    
    
    
#%% TESTING

#From phMELTs
Ol = np.array(pd.read_excel('test.xlsx',sheet_name='Ol'))
Cpx = np.array(pd.read_excel('test.xlsx',sheet_name='Cpx'))

# pred_KNN_Ol = minID_KNN.predict(Ol)
# pred_NN_Ol = minID_NN.predict(Ol)



#%% testing rellies work 
re = pd.read_excel('test.xlsx',sheet_name='rellie')
predre1 = minID_KNN.predict(re)
#predre2 = minID_LDA.predict(re)

#%%


