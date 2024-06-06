def emulate(newdata, emulator, rcm, topology, vars, outputFileName, type, predictand = "tas", years = None, scale = True, bias_correction = False, bias_correction_base = None):

    ### Load predictor data (.nc)
    x = xr.open_dataset(newdata)

    ### Load predictor base data (.nc)
    if type == 'PP-E':
        base_h   = xr.open_dataset('../data/predictors/upscaledrcm/x_' + emulator + '_historical_1996-2005.nc')
        base_85  = xr.open_dataset('../data/predictors/upscaledrcm/x_' + emulator + '_rcp85_2090-2099.nc')
        modelEmulator = emulator
    elif type == 'MOS-E':
        base_h   = xr.open_dataset('../data/predictors/gcm/x_' + emulator + '_historical_1996-2005.nc')
        base_85  = xr.open_dataset('../data/predictors/gcm/x_' + emulator + '_rcp85_2090-2099.nc')
        modelEmulator = emulator + '-' + rcm

    base = xr.concat([base_h,base_85], dim = 'time')

    if vars is not None:
        x = x[vars]
        base = base[vars]

    modelPath = '../models/' + topology + '-' + predictand + '-' + modelEmulator + '-' + type + '.h5'
    if years is not None:
        base  = xr.merge([base.sel(time = base['time.year'] == int(year)) for year in years])
        modelPath = '../models/' + topology + '-' + predictand + '-' + modelEmulator + '-' + type + '-year' + str(len(years)) + '.h5'

    ## Bias correction?..
    if bias_correction is True:
        print('bias correction...')
        base_h   = xr.open_dataset('../data/predictors/gcm/x_' + bias_correction_base + '_historical_1996-2005.nc')
        base_85  = xr.open_dataset('../data/predictors/gcm/x_' + bias_correction_base + '_rcp85_2090-2099.nc')
        base_gcm = xr.concat([base_h,base_85], dim = 'time')
        if vars is not None:
            base_gcm = base_gcm[vars]
        if years is not None:
            base_gcm  = xr.merge([base_gcm.sel(time = base_gcm['time.year'] == int(year)) for year in years])
        x = scaleGrid(x, base = base_gcm, ref = base, type = 'center', timeFrame = 'monthly', spatialFrame = 'gridbox')

    ## Scaling..
    if scale is True:
        print('scaling...')
        x = scaleGrid(x, base = base, type = 'standardize', spatialFrame = 'gridbox')

    ## Loading the cnn model...
    if predictand == 'tas':
        model = tf.keras.models.load_model(modelPath)
        description = {'description': 'air surface temperature (ºC)'}
    elif predictand == 'pr':
        model = tf.keras.models.load_model(modelPath, custom_objects = {'bernoulliGamma': bernoulliGamma})
        description = {'description': 'total daily precipitation (mm/day)'}


    ## Converting xarray to a numpy array and predict on the test set
    x_array = x.to_stacked_array("var", sample_dims = ["lon", "lat", "time"]).values
    pred = model.predict(x_array)

    ## Reshaping the prediction to a latitude-longitude grid
    mask = xr.open_dataset('../data/land_sea_mask/lsm_ald63.nc')
    if topology == 'deepesd':
        mask.sftlf.values[mask.sftlf.values == 0] = np.nan
        mask_Onedim = mask.sftlf.values.reshape((np.prod(mask.sftlf.shape)))
        ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 1]
        pred = reshapeToMap(grid = pred, ntime = x.dims['time'], nlat = mask.dims['y'], nlon = mask.dims['x'], indLand = ind)
    if topology == 'unet':
        sea = mask.sftlf.values == 0
        pred = np.squeeze(pred)
        pred[:,sea] = np.nan

    if predictand == 'pr':
        ## Loading the reference observation for the occurrence of precipitation ---------------------------
        gcm = newdata.split("_")[1].split("-")[0]
        yh = xr.open_dataset('../data/pr/pr_' + gcm + '-ald63_historical_1996-2005.nc') #, decode_times = False)
        y85 = xr.open_dataset('../data/pr/pr_' + gcm + '-ald63_rcp85_2090-2099.nc') #, decode_times = False)
        y = xr.concat([yh,y85], dim = 'time')
        y_bin = binaryGrid(y.pr, condition = 'GE', threshold = 1)
        ## -------------------------------------------------------------------------------------------------
        ## Prediction on the train set -----------
        base2 = scaleGrid(base, base = base,  type = 'standardize', spatialFrame = 'gridbox')
        ind_time = np.intersect1d(y.time.values, base2.time.values)
        base2 = base2.sel(time = ind_time)
        y = y.sel(time = ind_time)
        base_array = base2.to_stacked_array("var", sample_dims = ["lon", "lat", "time"]).values
        pred_ocu_train = model.predict(base_array)[:,:,0]
        pred_ocu_train = reshapeToMap(grid = pred_ocu_train, ntime = base2.dims['time'], nlat = mask.dims['y'], nlon = mask.dims['x'], indLand = ind)
        pred_ocu_train = xr.DataArray(pred_ocu_train, dims = ['time','lat','lon'], coords = {'lon': mask.x.values, 'lat': mask.y.values, 'time': y.time.values})
        ## ---------------------------------------
        ## Recovering the complete serie -----------
        pred = xr.Dataset(data_vars = {'p': (['time','lat','lon'], pred[:,:,:,0]),
                                       'log_alpha': (['time','lat','lon'], pred[:,:,:,1]),
                                       'log_beta': (['time','lat','lon'], pred[:,:,:,2])},
                                       coords = {'lon': mask.x.values, 'lat': mask.y.values, 'time': x.time.values})
        pred_bin = adjustRainfall(grid = pred.p, refPred = pred_ocu_train, refObs = y_bin)
        pred_amo = computeRainfall(log_alpha = pred.log_alpha, log_beta = pred.log_beta, bias = 1, simulate = True)
        pred = pred_bin * pred_amo
        pred = pred.values
        ## -----------------------------------------

    template_predictand = xr.open_dataset('../data/templates/template_predictand.nc')
    pred = xr.Dataset(
		data_vars = {predictand: (['time', 'y', 'x'], pred)},
	    coords = {'x': template_predictand.x.values[153:274], 'y': template_predictand.y.values[135:236], 'lon': (['y', 'x'],template_predictand.lon.values[135:236,153:274]), 'lat': (['y', 'x'], template_predictand.lat.values[135:236,153:274]), 'time': x.time.values},
	    attrs = description
	)
    print(outputFileName)
    pred.to_netcdf(outputFileName)
