import numpy as np
import xarray as xr
import icclim
import tensorflow as tf 

def openFiles(path):
	'''
	Open the files in the given path

	:param path: str, path to the files

	:return: xarray, files
	'''
	files = xr.open_mfdataset(path, combine='nested', concat_dim='time')

	return files


def loadMask(path):
	'''
	Load the land-sea mask from the given path and convert the 0-sea values to na-sea values 

	:param path: str, path to the land-sea mask

	:return: xarray, land-sea mask
    '''
	mask = xr.open_dataset(path)
	mask.sftlf.values[mask.sftlf.values == 0] = np.nan

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

	return grid


def applyMask(path, y, x, predictand):
	'''
	Apply the mask to the given grid

	:param grid: xarray, grid to be masked
	:param mask: xarray, mask

	:return: xarray, masked grid
	'''

	mask = loadMask(path)
	mask_Onedim = mask.sftlf.values.reshape((np.prod(mask.sftlf.shape)))
	ind = [i for i, value in enumerate(mask_Onedim) if value == 1]
	yTrain = y[predictand].values.reshape((x.dims['time'],np.prod(mask.sftlf.shape)))[:,ind]

	return yTrain

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

	## Train the model
	if predictand == 'tas' or predictand == 'tasmax' or predictand == 'tasmin':
		loss = 'mse'

	model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(lr = 0.0001))
	my_callbacks = [
		tf.keras.callbacks.EarlyStopping(patience = 30),
		tf.keras.callbacks.ModelCheckpoint(filepath = modelPath, monitor = 'val_loss', save_best_only = True)
	]
	model.fit(x = x, y = y, batch_size = 100, epochs = 10000, validation_split = 0.1, callbacks = my_callbacks)
	print(f'Model trained and save in: {modelPath}')


def time_comb(time):
	combinations = []
	# Loop through decades from the start year to the end year
	for start_year in range(time[0], time[1] + 1, 10):
		end_year = min(start_year + 9, time[1])  # Calculate the end year of the decade
		combination = f"{start_year}-{end_year}"  # Construct the combination string
		combinations.append(combination)  # Add the combination to the list
		if end_year == time[1]:
			break
	return combinations


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


def indexes_predictand(gcm, rcm, rcps, predictand, t_test):
    combinations = time_comb(t_test)

    if predictand == 'tasmin':
        index = 'TNn'  
    elif predictand == 'tasmax':
        index = 'TXx'
    else:
        raise ValueError('Predictand not valid')
    
    path_predictand = f'./data/predictand/'

    for rcp in rcps:
        for t in combinations:

            # Loading data               
            inputPredictand  = f'{path_predictand}{predictand}/{predictand}_{gcm}-{rcm}_{rcp}_{t}.nc'
            outputPredictand = f'{path_predictand}{index}/{index}_{gcm}-{rcm}_{rcp}_{t}.nc'

            icclim.index(
                in_files    = inputPredictand,
                out_file    = outputPredictand,
                slice_mode  = "month",
                index_name  = index 
            )
            print(outputPredictand)