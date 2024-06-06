##########################################################################
# Load packages
##########################################################################
import os
import xarray as xr
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import icclim                                   # for calculate TXx and TNn

### Loading functions from the functions folder
exec(open("./functions/buildEmulator.py").read())
exec(open("./functions/emulate.py").read())
exec(open("./functions/deepmodel.py").read())
exec(open("./functions/figure.py").read())
exec(open("./functions/auxiliaryFunctions.py").read())
exec(open("./functions/integratedGradients_vanilla.py").read())
exec(open("./functions/explainability.py").read())

##########################################################################################
# PARAMETER SETTING
##########################################################################################

# Predictor variables
vars = ['zg500', 'zg700',
'hus500', 'hus700', 'hus850',
'ua500', 'ua700', 'ua850',
'va500', 'va700', 'va850',
'ta500', 'ta700', 'ta850']

# RCM model (ALADIN63)
rcm = 'ald63'
# Network topology
topology = 'deepesd'
# GCM model (NorESM)
gcm = 'cnrm'
# path to predictors
path_predictors = './data/predictors/'
# path to predictions
path_predictions = './pred/'

# time periods
t_train = [2080, 2099]
t_test = [2010, 2099]

# Predictand variable
predictands = ['tas']


##########################################################################################
# BUILD THE EMULATORS
##########################################################################################
t_train = [2010, 2029]
# Calibration of statistical emulator (PP-E and MOS-E)
types = ['PP-E']
rcps = ['rcp26', 'rcp45','rcp85']
for type in types:
    for rcp in rcps:
            for predictand in predictands:
                    path_predictand = f'./data/predictand/{predictand}/'
                    buildEmulator(gcm = gcm,
                            rcm = rcm,
                            rcp = rcp,
                            vars = vars,
                            type = type,
                            t_train = t_train,
                            predictand = predictand,
                            topology = topology,
                            path_predictors = path_predictors,
                            path_predictand = path_predictand)

# types = ['MOS-E']
# rcps = ['rcp45','rcp85']
# for type in types:
#     for rcp in rcps:
#             for predictand in predictands:
#                     path_predictand = f'./data/predictand/{predictand}/'
#                     buildEmulator(gcm = gcm,
#                             rcm = rcm,
#                             rcp = rcp,
#                             vars = vars,
#                             type = type,
#                             t_train = t_train,
#                             predictand = predictand,
#                             topology = topology,
#                             path_predictors = path_predictors,
#                             path_predictand = path_predictand)

##########################################################################################
# BUILD THE EMULATORS
##########################################################################################
exec(open("./functions/emulate.py").read())
t_train = [2010, 2029]
rcps = ['rcp26','rcp45','rcp85']
for predictand in predictands:
        path_predictand = f'./data/predictand/{predictand}/'
        ## PP-E Perfect model: train predictors = RCM-U, test predictors = RCM-U.
        emulate(gcm = gcm,
                rcm = rcm,
                rcps = rcps,
                topology = topology,
                vars = vars,
                bias_correction = False,
                perfect = True,
                type = 'PP-E',
                t_train = t_train, 
                t_test = t_test, 
                predictand = predictand,
                path_predictors = path_predictors)
# predictands = ['tas','tasmin','tasmax','TNn','TXx']
rcps = ['rcp45','rcp85']
for predictand in predictands:
#         path_predictand = f'./data/predictand/{predictand}/'
#         ## MOS-E: train predictors = driving GCM, test predictors = driving GCM.
        emulate(gcm = gcm,
                rcm = rcm,
                rcps = rcps,
                topology = topology,
                vars = vars,
                bias_correction = False,
                type = 'MOS-E',
                t_train = t_train, 
                t_test = t_test, 
                predictand = predictand,
                path_predictors = path_predictors)
        ## PP-E GCM: train predictors = RCM-U, test predictors = driving GCM.
        emulate(gcm = gcm,
                rcm = rcm,
                rcps = rcps,
                topology = topology,
                vars = vars,
                bias_correction = False,
                type = 'PP-E',
                t_train = t_train, 
                t_test = t_test, 
                predictand = predictand,
                path_predictors = path_predictors)
        ## PP-E GCM-BC: train predictors = RCM-U, test predictors = driving GCM bias adjusted.
        emulate(gcm = gcm,
                rcm = rcm,
                rcps = rcps,
                topology = topology,
                vars = vars,
                bias_correction = True,
                type = 'PP-E',
                t_train = t_train, 
                t_test = t_test, 
                predictand = predictand,
                path_predictors = path_predictors)

# ##########################################################################################
# # FIGURES
# ##########################################################################################
exec(open("./functions/misFiguras.py").read())
# metrics of evaluation
# metrics = ['bias', 'mean', 'P02', 'P98', 'rmse']
predictands = ['tas']
metrics = ['bias','rmse','P02','P98']
times = [[2010,2029],[2040,2059],[2060,2079],[2080,2099]]
rcps = ['rcp26','rcp45','rcp85']
figsize = (36, 18)
types = ['PP-E']
for type_ in types:
        for predictand in predictands:
                path_predictand = f'./data/predictand/{predictand}/'
                figures2(gcm, rcm, rcps, metrics, t_test, figsize, predictand, type_, 
                        topology, times, path_predictand, path_predictions, BC = False, perfect = True)
print('done')
# # #############################################################################################
# # # TNn
# # #############################################################################################
import icclim
exec(open("script_TNn.py").read())

# from script_TNn.py import TNn_a, TNn
 
# rcps = ['rcp26', 'rcp45', 'rcp85']
# TNn_a(gcm, rcm, rcps, predictand, type_, topology, t_test, BC = False)

exec(open("./functions/misFiguras.py").read())
gcm = 'cnrm'
rcm = 'ald63'
times = [(2010,2029),(2040,2059),(2060,2079),(2080,2099)]
rcps = ['rcp26', 'rcp45', 'rcp85']  # Only two RCPs now
path_predictand = './data/predictand/tas/'
predictand = 'tas'
figsize = (30, 20)  # Adjust figsize to accommodate more subplots
for type in types:
        panel(gcm, rcm, rcps, type, times, path_predictand, predictand, figsize, perfect=False, BC = False)