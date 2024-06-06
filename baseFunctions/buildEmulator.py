def buildEmulator(gcm, rcm, predictand, vars, type, topology, years = None, scale = True):

	### Load predictor data (.nc)
	if type == 'PP-E':
		training_dataset = 'upscaledrcm'
		xh = xr.open_dataset('./data/' + training_dataset + '/x_' + gcm + '-' + rcm + '_historical_1996-2005.nc')
		x85 = xr.open_dataset('./data/' + training_dataset + '/x_' + gcm + '-' + rcm + '_rcp85_2090-2099.nc')
	elif type == 'MOS-E':
		training_dataset = 'gcm'
		xh = xr.open_dataset('./data/' + training_dataset + '/x_' + gcm + '_historical_1996-2005.nc')
		x85 = xr.open_dataset('./data/' + training_dataset + '/x_' + gcm + '_rcp85_2090-2099.nc')

	x = xr.concat([xh,x85], dim = 'time')
	if vars is not None:
		x = x[vars]

	modelPath = './models/' + predictand + '/' + topology + '-' + gcm + '-' + rcm + '-' + type + '.h5'
	if years is not None:
		x  = xr.merge([x.sel(time = x['time.year'] == int(year)) for year in years])
		modelPath = './models/' + predictand + '/' + topology + '-' + gcm + '-' + rcm + '-' + type + '-year' + str(len(years)) + '.h5'
	### Scaling..
	if scale is True:
		x = scaleGrid(x, base = x, type = 'standardize', spatialFrame = 'gridbox')

	### Loading predictand data (.nc)
	yh = xr.open_dataset('./data/' + predictand + '/' + predictand + '_' + gcm + '-' + rcm + '_historical_1996-2005.nc')
	y85 = xr.open_dataset('./data/' + predictand + '/' + predictand + '_' + gcm + '-' + rcm + '_rcp85_2090-2099.nc')
	y = xr.concat([yh,y85], dim = 'time')
	if predictand == 'pr':
		y = binaryGrid(y, condition = 'GE', threshold = 1, partial = True)

	## Converting xarray to a numpy array
	ind_time = np.intersect1d(y.time.values, x.time.values)
	x = x.sel(time = ind_time)
	y = y.sel(time = ind_time)
	x_array = x.to_stacked_array("var", sample_dims = ["lon", "lat", "time"]).values

	outputShape = None
	### Mask the sea in the deepesd model
	mask = xr.open_dataset('./data/lsm/lsm_ald63.nc')
	if topology == 'deepesd':
		mask.sftlf.values[mask.sftlf.values == 0] = np.nan
		mask_Onedim = mask.sftlf.values.reshape((np.prod(mask.sftlf.shape)))
		ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 1]
		yTrain = y[predictand].values.reshape((x.dims['time'],np.prod(mask.sftlf.shape)))[:,ind]
		if predictand == 'pr':
			yTrain = yTrain - 0.99
			yTrain[yTrain < 0] = 0

		outputShape = yTrain.shape[1]
	if topology == 'unet':
		sea = mask.sftlf.values == 0
		y[predictand].values[:,sea] = 0
		yTrain = y[predictand].values
		outputShape = None

	### Define deep model
	model = deepmodel(topology = topology,
	                  predictand = predictand,
	                  inputShape = x_array.shape[1::],
					  outputShape = outputShape)

	### Train the model
	if predictand == 'tas':
		loss = 'mse'
	elif predictand == 'pr':
		loss = bernoulliGamma
	model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(lr = 0.0001))
	my_callbacks = [
    	tf.keras.callbacks.EarlyStopping(patience = 30),
    	tf.keras.callbacks.ModelCheckpoint(filepath = modelPath, monitor = 'val_loss', save_best_only = True)
	]
	model.fit(x = x_array, y = yTrain, batch_size = 100, epochs = 10000, validation_split = 0.1, callbacks = my_callbacks)
