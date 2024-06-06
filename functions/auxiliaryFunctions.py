def bernoulliGamma(true, pred):
    '''
    Custom loss function used to reproduce the distribution of precipitations
    By minimizing this function we are minimizing the negative log-likelihood
    of a Bernoulli-Gamma distribution
    (Tensorflow 2.X.X)
    '''

    ocurrence = pred[:,:,0] # p parameter
    shape_parameter = K.exp(pred[:,:,1]) # shape parameter
    scale_parameter = K.exp(pred[:,:,2]) # beta parameter

    bool_rain = K.cast(K.greater(true, 0), 'float32')
    epsilon = 0.000001 # avoid nan values

    noRainCase = (1 - bool_rain) * K.log(1 - ocurrence + epsilon) # no rain
    rainCase = bool_rain * (K.log(ocurrence + epsilon) + # rain
                           (shape_parameter - 1) * K.log(true + epsilon) -
                           shape_parameter * K.log(scale_parameter + epsilon) -
                           tf.math.lgamma(shape_parameter + epsilon) -
                           true / (scale_parameter + epsilon))

    return - K.mean(noRainCase + rainCase) # loss output

# def rmse(obs, prd, var = 'tas'):
#     rmse_values = np.sqrt(np.square(prd[var].values -  obs[var].values).mean(axis = 0))
#     template = prd[var].mean('time')
#     template.values = rmse_values
#     return template

def biasR01(obs, prd, var, period = None):
    if period == 'winter':
        obs2 = xr.merge([obs.sel(time = obs['time.month'] == z) for z in [1,2,12]])
        prd2 = xr.merge([prd.sel(time = prd['time.month'] == z) for z in [1,2,12]])
    if period == 'summer':
        obs2 = xr.merge([obs.sel(time = obs['time.month'] == z) for z in [6,7,8]])
        prd2 = xr.merge([prd.sel(time = prd['time.month'] == z) for z in [6,7,8]])
    if period == None:
        obs2 = obs
        prd2 = prd
    obs2_bin = binaryGrid(obs2, condition = "GE", threshold = 1).mean('time')
    prd2_bin = binaryGrid(prd2, condition = "GE", threshold = 1).mean('time')
    bias = (prd2_bin - obs2_bin) / obs2_bin
    return bias

def biasSDII(obs, prd, var, period = None):
    if period == 'winter':
        obs2 = xr.merge([obs.sel(time = obs['time.month'] == z) for z in [1,2,12]])
        prd2 = xr.merge([prd.sel(time = prd['time.month'] == z) for z in [1,2,12]])
    if period == 'summer':
        obs2 = xr.merge([obs.sel(time = obs['time.month'] == z) for z in [6,7,8]])
        prd2 = xr.merge([prd.sel(time = prd['time.month'] == z) for z in [6,7,8]])
    if period == None:
        obs2 = obs
        prd2 = prd
    obs2_sdii = obs2.where(obs2 >= 1).mean('time')
    prd2_sdii = prd2.where(prd2 >= 1).mean('time')
    bias = (prd2_sdii - obs2_sdii) / obs2_sdii
    return bias

def biasP98Wet(obs, prd, var , period = None):
    if period == 'winter':
        obs2 = xr.merge([obs.sel(time = obs['time.month'] == z) for z in [1,2,12]])
        prd2 = xr.merge([prd.sel(time = prd['time.month'] == z) for z in [1,2,12]])
    if period == 'summer':
        obs2 = xr.merge([obs.sel(time = obs['time.month'] == z) for z in [6,7,8]])
        prd2 = xr.merge([prd.sel(time = prd['time.month'] == z) for z in [6,7,8]])
    if period == None:
        obs2 = obs
        prd2 = prd
    obs2_p98 = obs2.where(obs2 >= 1).quantile(0.98, 'time')
    prd2_p98 = prd2.where(prd2 >= 1).quantile(0.98, 'time')
    bias = (prd2_p98 - obs2_p98) / obs2_p98
    return bias




def loadMask(path,rcm):
	# Loading data
	mask = xr.open_dataset(path)

	# Change arome's variable name
	if rcm == 'arm':
		mask = mask.rename({"var1" : "sftlf"})

	# Converting 0-sea values to na-sea values
	mask.sftlf.values[mask.sftlf.values == 0] = np.nan
	return mask

def adjustRainfall(grid, refPred, refObs):
	frequencyOfNoRain = 1 - refObs.mean(axis = 0)
	for i in range(grid.shape[1]):
		for j in range(grid.shape[2]):
			if np.isnan(frequencyOfNoRain[i,j]):
				grid[:,i,j]	= np.nan
			else:
				threshold = np.quantile(refPred[:,i,j], frequencyOfNoRain[i,j])
				grid[:,i,j] = binaryGrid(grid[:,i,j], condition = 'GE', threshold = threshold)
	return grid

def reshapeToMap(grid, ntime, nlat, nlon, indLand):
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

def binaryGrid(grid, condition, threshold, partial = False, values = [0,1]):

	### Numpy arrays
	# if condition == 'GE':
	# 	grid = grid.where(grid >= threshold, values[1], values[0])
	# if condition == 'LE':
	#	grid = grid.where(grid <= threshold, values[1], values[0])
	# if condition == 'LT':
	#	grid = grid.where(grid < threshold, values[1], values[0])
	# if condition == 'GT':
	#	grid = grid.where(grid > threshold, values[1], values[0])

    ### xarrays (xDataArray non xDatasets)
	mask = xr.where(grid.isnull(), np.nan, 1)
	if condition == 'GE':
		if partial is True:
			grid = xr.where(grid < threshold, values[0], grid)
		else:
			grid = xr.where(grid < threshold, values[0], values[1])
	if condition == 'GT':
		if partial is True:
			grid = xr.where(grid <= threshold, values[0], grid)
		else:
			grid = xr.where(grid <= threshold, values[0], values[1])

	return grid * mask

def computeRainfall(log_alpha, log_beta, bias = None, simulate = False):
	alpha = np.exp(log_alpha)
	beta = np.exp(log_beta)
	if simulate is False:
		grid = alpha * beta
	elif simulate is True:
		# bernoulliParams = np.stack((alpha, beta), axis =  2)
		# grid = np.apply_along_axis(lambda x: np.random.gamma(shape = x[0], scale = x[1], size = 1), 2, bernoulliParams)
		# grid = da.array.random.gamma(shape = alpha, scale = beta)
		values = np.random.gamma(shape = alpha.values, scale = beta.values)
		grid = xr.DataArray(values, dims = log_alpha.dims ,coords = log_alpha.coords)
	if bias is not None:
		grid = grid + bias
	return grid

def reshapeToMap(grid, ntime, nlat, nlon, indLand):
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


def scaleGrid(grid, base = None, ref = None, timeFrame = None, spatialFrame = 'gridbox', type = 'center'):

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

def normalize(grid, base = None):
    if base == None:
        base = grid
    m1 = base.max('time')
    m2 = base.min('time')
    grid = (grid - m2) / (m1 - m2)
    return grid

def paths(gcm, rcm, rcp, type = None, t = None, training_dataset = None, BC = None, predictand = None, topology = None, path_predictors = None, path_predictand = None, perfect = None, case = None):
	if case == 'buildEmulator':
		if path_predictors is not None and type == 'PP-E':
			path = f'{path_predictors}{training_dataset}/x_{gcm}-{rcm}_{rcp}_{t}.nc'
		if path_predictors is not None and type == 'MOS-E':
			path = f'{path_predictors}{training_dataset}/x_{gcm}_{rcp}_{t}.nc'
		if path_predictand is not None:
			path = f'{path_predictand}{predictand}_{gcm}-{rcm}_{rcp}_{t}.nc'
	if case == 'perfect':
		path = f'{path_predictors}{training_dataset}/x_{gcm}-{rcm}_{rcp}_{t}.nc'
	if case == 'gcm':
		path = f'{path_predictors}{training_dataset}/x_{gcm}_{rcp}_{t}.nc'
	if case == 'models':
		path = f'./models/{predictand}/{topology}-{predictand}-{type}-{gcm}-{rcm}-{rcp}.h5'
	if case == 'predictions' and BC is False and perfect is False:
		path = f'{path_predictions}{predictand}/{type}_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp[0]}_test-{rcp[1]}_{t[0]}-{t[1]}.nc'
	if case == 'predictions' and BC is False and perfect is True:
		path = f'{path_predictions}{predictand}/{type}-perfect_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp[0]}_test-{rcp[1]}_{t[0]}-{t[1]}.nc'
	if case == 'predictions' and BC is True: 
		path = f'{path_predictions}{predictand}/{type}-BC_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp[0]}_test-{rcp[1]}_{t[0]}-{t[1]}.nc'
	return path

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