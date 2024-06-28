from buildEmulator import buildEmulator
from emulate import emulate
import sys

# Variables
# General variables
vars = ['zg500', 'zg700',
        'hus500', 'hus700', 'hus850',
        'ua500', 'ua700', 'ua850',
        'va500', 'va700', 'va850',
        'ta500', 'ta700', 'ta850']
predictand = 'tas'
gcm = 'cnrm'
rcm = 'ald63'
topology = 'deepesd'
aprox = 'PP-E'

predictorsUpscaledPath = '../../data/predictors/upscaledrcm/'
predictorsGCMPath = '../../data/predictors/gcm/'
predictandPath = f'../../data/predictand/{predictand}/'
modelPath = '../../m2/'
predictionPath = '../../p2/'

rcp_std = 'rcp45'
rcp_train = 'rcp45'

t_train = [2080, 2099]
t_base = [2080, 2099]

# Auxiliar funtions to generate the paths of the files

def timeCombinations(time):
    '''
    Create the time combinations strings

    :param time: list, time period [start, end] in years

    :return: list, time combinations strings
    '''
    combinations = []
	# Loop through decades from the start year to the end year
    for start_year in range(time[0], time[1] + 1, 10):
        end_year = min(start_year + 9, time[1])  # Calculate the end year of the decade
        combination = f"{start_year}-{end_year}"  # Construct the combination string
        combinations.append(combination)  # Add the combination to the list
        if end_year == time[1]:
            break
    return combinations

def filePaths(time, path, rcp, gcm, rcm, predictand):
    '''
    Generate the paths of the files to be used in the emulator
    
    :param time: list, time period
    :param path: str, path to the predictors data
    :param rcp: str, rcp scenario
    
    :return: list, paths of the files
    '''

    periods = timeCombinations(time)
    paths = []
    for period in periods:
        if path == '../../data/predictors/upscaledrcm/':
            paths.append(f'{path}x_{gcm}-{rcm}_{rcp}_{period}.nc')
        elif path == '../../data/predictors/gcm/':
            paths.append(f'{path}x_{gcm}_{rcp}_{period}.nc')
        else:
            paths.append(f'{path}{predictand}_{gcm}-{rcm}_{rcp}_{period}.nc')
    return paths

def modelPaths(path, rcp, time, predictand, aprox, gcm, rcm):
    '''
    Generate the path of the model
    
    :param modelRootPath: str, path to the model
    :param rcp: str, rcp scenario
    :param t_train: list, time period of the training data
    
    :return: str, path of the model
    '''

    return f'{path}deepesd-{predictand}-{aprox}-{gcm}-{rcm}-{rcp}_{time[0]}-{time[1]}.h5'

def predictionPaths(path, rcp_train, rcp_test, t_train, t_test, predictand, aprox, gcm, rcm, BC = False, perfect = False):
    '''
    Generate the path of the prediction
    
    :param path: str, path to the prediction
    :param rcp: str, rcp scenario
    :param time: list, time period of the prediction
    :param predictand: str, predictand
    :param aprox: str, approximation
    :param gcm: str, gcm model
    :param rcm: str, rcm model

    :return: str, path of the prediction
    '''
    if aprox == 'MOS-E':
        predPath = f'{path}{aprox}_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
    elif aprox == 'PP-E' and BC == False and perfect == False:
        predPath = f'{path}{aprox}_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
    elif aprox == 'PP-E' and BC == True and perfect == False:
        predPath = f'{path}/{aprox}_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
    elif aprox == 'PP-E' and BC == False and perfect == True:
        predPath = f'{path}/{aprox}_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
    else:
        print('Invalid approximation')
        sys.stdout.flush()
    return predPath


def descriptions(predictand):
    '''
    Generate the description of the prediction
    
    :param predictand: str, predictand
    
    :return: str, description of the prediction
    '''
    if predictand == 'tas':
        return {'description': 'air surface temperature (ºC)'}
    elif predictand == 'tasmin':
        return {'description': 'minimum air surface temperature (ºC)'}
    elif predictand == 'tasmax':
        return {'description': 'maximum air surface temperature (ºC)'}
    else:
        print('Invalid predictand')
        sys.stdout.flush()

# Generate the paths of the files    
predictorsFileTrain = filePaths(t_train, predictorsUpscaledPath, rcp_train, gcm, rcm, predictand)
baseFile = filePaths(t_base, predictorsUpscaledPath, rcp_std, gcm, rcm, predictand)
predictandFile = filePaths(t_train, predictandPath, rcp_train, gcm, rcm, predictand)
modelFile = modelPaths(modelPath, rcp_train, t_train, predictand, aprox, gcm, rcm)
maskFile = '../../data/land_sea_mask/lsm_ald63.nc'

# Build the emulator
buildEmulator(predictorsPath = predictorsFileTrain,
              basePath = baseFile,
              predictandPath = predictandFile,
              modelPath = modelFile,
              maskPath = maskFile,
              topology = topology,
              predictand = predictand,
              vars = vars,
              scale = True)

# Emulate the predictand
rcp_test = 'rcp85'
t_test = [2010, 2099]
BiasCorrection = False
perfect = False
predictorsFileTest = filePaths(t_test, predictorsGCMPath, rcp_test, gcm, rcm, predictand)
baseGCMPath = filePaths(t_test, predictorsGCMPath, rcp_test, gcm, rcm, predictand)
baseRefPath = filePaths(t_test, predictorsUpscaledPath, rcp_test, gcm, rcm, predictand)

predictionFile = predictionPaths(predictionPath, rcp_train, rcp_test, t_train, t_test, predictand, aprox, gcm, rcm, BiasCorrection, perfect)
description = descriptions(predictand)

emulate(predictionFile, predictorsFileTest, baseFile, modelFile, maskFile, description,
        predictand, vars, scale = True, BC = BiasCorrection, baseGCMPath = baseGCMPath, baseRefPath = baseRefPath)


from auxiliaryFunctions import openFiles 
import numpy as np
import matplotlib.pyplot as plt

predPath = ['../../p2/PP-E_tas_deepesd_alp12_cnrm-ald63_train-rcp45-2080-2099_test-rcp85-2010-2099.nc']
predPathOG = ['../../pred/tas/PP-E_tas_deepesd_alp12_cnrm-ald63_train-rcp45-2080-2099_test-rcp85-2010-2099.nc']

pred = openFiles(predPath)
predOG = openFiles(predPathOG)

tasPath = ['../../data/predictand/tas/tas_cnrm-ald63_rcp85_2080-2089.nc', 
           '../../data/predictand/tas/tas_cnrm-ald63_rcp85_2090-2099.nc']
tas = openFiles(tasPath).sel(time = slice('2080', '2099'))


pred = pred.sel(time = slice('2080', '2099'))
predOG = predOG.sel(time = slice('2080', '2099'))

bias1 = np.mean(pred['tas'].values, axis = 0) - np.mean(tas['tas'].values, axis = 0)
template = pred['tas'].mean('time')
template.values = bias1
plt.figure(); template.plot(vmin=-2, vmax=2,cmap = 'RdBu_r'); plt.savefig('./check2.png')   

bias2 = np.mean(predOG['tas'].values, axis = 0) - np.mean(tas['tas'].values, axis = 0)
templateReal = predOG['tas'].mean('time')
templateReal.values = bias2
plt.figure(); templateReal.plot(vmin=-2, vmax=2,cmap = 'RdBu_r'); plt.savefig('./checkReal2.png')