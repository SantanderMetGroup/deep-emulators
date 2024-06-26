import numpy as np
import xarray as xr
import icclim
import tensorflow as tf 
import sys 
import time
import os

def openFiles(path):
	'''
	Open the files in the given path

	:param path: str, path to the files

	:return: xarray, files
	'''
	# # files = xr.open_mfdataset(path, combine='nested', concat_dim='time')
	# files = xr.open_mfdataset(path, combine='nested', concat_dim='time', chunks={'time': 365, 'lat': 107, 'lon': 191}, parallel=True)
	# print("Opened files from:", path)
	# sys.stdout.flush()

	# Load files without specifying chunk sizes or with different chunking
	files = xr.open_mfdataset(path, combine='nested', concat_dim='time', parallel=True)
    # Rechunk the dataset after loading
	files = files.chunk({'time': 365, 'lat': 107, 'lon': 191})
	print("Opened and rechunked files from:", path)
	sys.stdout.flush()
	return files


def loadMask(path):
	'''
	Load the land-sea mask from the given path and convert the 0-sea values to na-sea values 

	:param path: str, path to the land-sea mask

	:return: xarray, land-sea mask
    '''
	mask = xr.open_dataset(path)
	mask.sftlf.values[mask.sftlf.values == 0] = np.nan
	print("Loaded mask from:", path)
	sys.stdout.flush()

	return mask


def scaleGrid(grid, base = None, ref = None, timeFrame = None, spatialFrame = 'gridbox', type = 'center'):
	'''
	Scale the grid

	:param grid: xarray, grid to be scaled
	:param base: xarray, grid to be used as base for the scaling
	:param ref: xarray, grid to be used as reference for the scaling
	:param timeFrame: str, time frame for the scaling
	:param spatialFrame: str, spatial frame for the scaling
	:param type: str, type of scaling

	:return: xarray, scaled grid
	'''
	print("Starting scaling...")
	sys.stdout.flush()
	if base is None:
		base = grid

	## Selecting the dimension along which the scaling is to be done
	if spatialFrame == 'gridbox':
		dimension = 'time'
	if spatialFrame == 'field':
		dimension = ['lat', 'lon']

	## Scaling...
	if timeFrame is None:
		if type == 'center':
			if ref is None:
				grid = grid - base.mean(dimension)
			if ref is not None:
				grid = grid - base.mean(dimension) + ref.mean(dimension)
		if type == 'standardize':
			if ref is None:
				grid = (grid - base.mean(dimension)) / base.std(dimension)
			if ref is not None:
				grid = (grid - base.mean(dimension)) / base.std(dimension) * ref.std(dimension) + ref.mean(dimension)

	## Scaling with monthly statistics...
	if timeFrame == 'monthly':
		baseMonths = base.groupby('time.month')
		months = baseMonths.groups.keys()
		if type == 'center':
			if ref is None:
				grid = xr.merge([grid.sel(time = grid['time.month'] == z) - baseMonths.mean('time').sel(month = z).drop_vars('month') for z in months])
			if ref is not None:
				refMonths = ref.groupby('time.month')
				grid = xr.merge([grid.sel(time = grid['time.month'] == z) - baseMonths.mean('time').sel(month = z).drop_vars('month') + refMonths.mean('time').sel(month = z).drop_vars('month') for z in months])
		if type == 'standardize':
			if ref is None:
				grid = xr.merge([(grid.sel(time = grid['time.month'] == z) - baseMonths.mean('time').sel(month = z).drop_vars('month')) / baseMonths.std('time').sel(month = z).drop_vars('month') for z in months])
			if ref is not None:
				refMonths = ref.groupby('time.month')
				grid = xr.merge([(grid.sel(time = grid['time.month'] == z) - baseMonths.mean('time').sel(month = z).drop_vars('month')) / baseMonths.std('time').sel(month = z).drop_vars('month') * refMonths.std('time').sel(month = z).drop_vars('month') + refMonths.mean('time').sel(month = z).drop_vars('month') for z in months])

	print("Completed")
	sys.stdout.flush()
	return grid


def applyMask(path, y, x, predictand, case):
	'''
	Apply the mask to the given grid

	:param grid: xarray, grid to be masked
	:param mask: xarray, mask

	:return: xarray, masked grid
	'''

	mask = loadMask(path)
	mask_Onedim = mask.sftlf.values.reshape((np.prod(mask.sftlf.shape)))
	ind = [i for i, value in enumerate(mask_Onedim) if value == 1]
	if case == 'emulate':
		y_reshaped = reshapeToMap(grid = y, ntime = x.dims['time'], nlat = mask.dims['lat'], nlon = mask.dims['lon'], indLand = ind)
	elif case == 'buildEmulator':	
		y_reshaped = y[predictand].values.reshape((x.sizes['time'],np.prod(mask.sftlf.shape)))[:,ind]
	print("Applied mask to the data")
	sys.stdout.flush()
	return y_reshaped

def reshapeToMap(grid, ntime, nlat, nlon, indLand):
	'''
	Reshape the grid to a latitude-longitude grid

	:param grid: numpy, grid to be reshaped
	:param ntime: int, number of time steps
	:param nlat: int, number of latitude points
	:param nlon: int, number of longitude points
	:param indLand: list, indices of the land points

	:return: numpy, reshaped grid
	'''
	if len(grid.shape) == 2:
		grid = np.expand_dims(grid, axis = 2)
		vars = 1
	InnerList=[]
	for i in range(grid.shape[2]):
		p = np.full((ntime,nlat*nlon), np.nan)
		p[:,indLand] = grid[:,:,i]
		p = p.reshape((ntime,nlat,nlon,1))
		InnerList.append(p)
	grid = np.concatenate(InnerList, axis = 3)
	return grid.squeeze()

def saveTrainingTime(savePath, modelPath, elapsed_time):
    '''
    Save the training time to a file

    :param modelPath: str, path of the trained model
    :param elapsed_time: float, time taken to train the model in seconds

    :return: None
    '''
    training_log_path = savePath  # Specify your log file name/path
    with open(training_log_path, 'a') as file:
        file.write(f'Model: {modelPath}, Training Time: {elapsed_time:.2f} seconds, Date: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')


def trainModel(x, y, model, modelPath, predictand):
	'''
	Train the model

	:param x: numpy, predictors
	:param y: xarray, predictand
	:param model: keras model, model
	:param modelPath: str, path to save the model
	:param predictand: str, predictand

	:return: None
	'''
    # Record the start time
	start_time = time.time()

	# Train the model
	if predictand == 'tas' or predictand == 'tasmax' or predictand == 'tasmin':
		loss = 'mse'

	model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001))
	my_callbacks = [
		tf.keras.callbacks.EarlyStopping(patience = 30),
		tf.keras.callbacks.ModelCheckpoint(filepath = modelPath, monitor = 'val_loss', save_best_only = True)
	]
	model.fit(x = x, y = y, batch_size = 100, epochs = 10000, validation_split = 0.1, callbacks = my_callbacks)
    # Calculate the elapsed time
	elapsed_time = time.time() - start_time

	print(f'Model trained and save in: {modelPath}')
	sys.stdout.flush()

	savePath = './training_time.txt'
	# Save the training time using the new function
	saveTrainingTime(savePath, modelPath, elapsed_time)

def biasCorrection(x, baseGCMPath, baseRefPath, vars = None):
	'''
	Bias correction

	:param x: xarray, data to be bias corrected
	:param baseGCMPath: str, path to the base GCM data (train period and rcp)
	:param baseRefPath: str, path to the base UpscaledRCM data (train period and rcp)
	:param vars: list, variables to be bias corrected
	
	:return: xarray, bias corrected data
	'''

	baseGCM = openFiles(baseGCMPath)
	baseRef = openFiles(baseRefPath)

	if vars is not None:
		baseGCM = baseGCM[vars]
		baseRef = baseRef[vars]
	
	return scaleGrid(x, base = baseGCM, ref = baseRef, type = 'center', timeFrame = 'monthly', spatialFrame = 'gridbox')

def createDataset(pred, x, description, predictand, maskPath):
	'''
	Create a xarray dataset with the prediction

	:param pred: numpy, prediction
	:param x: xarray, predictors
	:param description: str, description of the prediction
	:param predictand: str, predictand

	:return: xarray, prediction
	'''

	mask = loadMask(maskPath)
	
	pred = xr.Dataset(
    data_vars = {predictand: (['time', 'lat', 'lon'], pred)},
    coords = {'lon': mask.lon.values, 
                'lat': mask.lat.values, 
                'time': x.time.values},
    attrs = {'description': f'{description}'}
    )
	return pred	

def indexes_predicions(gcm, rcm, rcps, predictand, type, topology, t_train, t_test, BC, perfect):

    if predictand == 'tasmin':
        index = 'TNn'  
    elif predictand == 'tasmax':
        index = 'TXx'
    else:
        raise ValueError('Predictand not valid')
    
    for rcp_test in rcps:
        for rcp_train in rcps:

            # Loading data
            if type == 'MOS-E':
                inputFileName = f'pred/{predictand}/MOS-E_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
                outputFileName = f'pred/{index}/MOS-E_{index}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
            elif BC and not perfect:
                inputFileName = f'pred/{predictand}/PP-E-BC_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
                outputFileName = f'pred/{index}/PP-E-BC_{index}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
            elif not BC and not perfect:
                inputFileName = f'pred/{predictand}/PP-E_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
                outputFileName = f'pred/{index}/PP-E_{index}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
            elif perfect: 
                inputFileName = f'pred/{predictand}/PP-E-perfect_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'
                outputFileName = f'pred/{index}/PP-E-perfect_{index}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc'

            icclim.index(
                in_files    = inputFileName,
                out_file    = outputFileName,
                slice_mode  = "month",
                index_name  = index 
            )
            print(outputFileName)


# def indexes_predictand(gcm, rcm, rcps, predictand, t_test):
#     combinations = time_comb(t_test)

#     if predictand == 'tasmin':
#         index = 'TNn'  
#     elif predictand == 'tasmax':
#         index = 'TXx'
#     else:
#         raise ValueError('Predictand not valid')
    
#     path_predictand = f'./data/predictand/'

#     for rcp in rcps:
#         for t in combinations:

#             # Loading data               
#             inputPredictand  = f'{path_predictand}{predictand}/{predictand}_{gcm}-{rcm}_{rcp}_{t}.nc'
#             outputPredictand = f'{path_predictand}{index}/{index}_{gcm}-{rcm}_{rcp}_{t}.nc'

#             icclim.index(
#                 in_files    = inputPredictand,
#                 out_file    = outputPredictand,
#                 slice_mode  = "month",
#                 index_name  = index 
#             )
#             print(outputPredictand)