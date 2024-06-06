def buildEmulator(gcm, rcm, rcp, predictand, vars, type, t_train, topology, path_predictors, 
				  path_predictand, scale = True):

	t_train = time_comb(t_train)

	### Load predictor data (.nc)
	if type == 'PP-E':
		training_dataset = 'upscaledrcm'
		files = [paths(gcm, rcm, rcp, type, t, training_dataset, path_predictors = path_predictors, case ='buildEmulator') for t in t_train]
		[print("Predictor data: ", file) for file in files]
		x = xr.open_mfdataset(files, combine='nested', concat_dim='time')
		# same base for all datasets (rcp 45 in 2080-2099)
		path_base = [f'./data/predictors/{training_dataset}/x_cnrm-ald63_rcp45_{t}.nc' for t in ['2080-2089', '2090-2099']]
		[print("Predictor data base: ", file) for file in path_base]
		base = xr.open_mfdataset(path_base, combine='nested', concat_dim='time')

	elif type == 'MOS-E':
		training_dataset = 'gcm'
		files = [paths(gcm, rcm, rcp, type, t, training_dataset, path_predictors = path_predictors, case ='buildEmulator') for t in t_train]
		[print("Predictor data: ", file) for file in files]
		x = xr.open_mfdataset(files, combine='nested', concat_dim='time')
		# same base for all datasets (rcp 45 in 2080-2099)
		path_base = [f'./data/predictors/{training_dataset}/x_cnrm_rcp45_{t}.nc' for t in ['2080-2089', '2090-2099']]
		[print("Predictor data base: ", file) for file in path_base]
		base = xr.open_mfdataset(path_base, combine='nested', concat_dim='time')

	if vars is not None:
		x = x[vars]

	# modelPath = paths(gcm, rcm, rcp, type, predictand= predictand, topology = topology, path_predictors = path_predictors, case ='models')
	modelPath = f'./models/{predictand}/{topology}-{predictand}-{type}-{gcm}-{rcm}-{rcp}_2010-2029.h5'
	
	### Scaling..
	if scale:
		x = scaleGrid(x, base = base, type = 'standardize', spatialFrame = 'gridbox')

	### Loading predictand data (.nc)
	files = [paths(gcm, rcm, rcp, type, t, training_dataset, predictand = predictand, path_predictand = path_predictand, case ='buildEmulator') for t in t_train]
	y = xr.open_mfdataset(files, combine='nested', concat_dim='time')

	## Converting xarray to a numpy array
	ind_time = np.intersect1d(y.time.values, x.time.values)
	x = x.sel(time = ind_time)
	y = y.sel(time = ind_time)
	x_array = x.to_stacked_array("var", sample_dims = ["lon", "lat", "time"]).values

	outputShape = None
	### Mask the sea in the deepesd model
	mask = xr.open_dataset('./data/land_sea_mask/lsm_ald63.nc')
	if topology == 'deepesd':
		mask.sftlf.values[mask.sftlf.values == 0] = np.nan
		mask_Onedim = mask.sftlf.values.reshape((np.prod(mask.sftlf.shape)))
		ind = [i for i, value in enumerate(mask_Onedim) if value == 1]
		yTrain = y[predictand].values.reshape((x.dims['time'],np.prod(mask.sftlf.shape)))[:,ind]
		outputShape = yTrain.shape[1]

	### Define deep model
	model = deepmodel(topology = topology,
					  predictand = predictand,
					  inputShape = x_array.shape[1::],
					  outputShape = outputShape)

	### Train the model
	if predictand == 'tas' or predictand == 'tasmax' or predictand == 'tasmin':
		loss = 'mse'

	model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(lr = 0.0001))
	my_callbacks = [
		tf.keras.callbacks.EarlyStopping(patience = 30),
		tf.keras.callbacks.ModelCheckpoint(filepath = modelPath, monitor = 'val_loss', save_best_only = True)
	]
	model.fit(x = x_array, y = yTrain, batch_size = 100, epochs = 10000, validation_split = 0.1, callbacks = my_callbacks)
	print(modelPath)