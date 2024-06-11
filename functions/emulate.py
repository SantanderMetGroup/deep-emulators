def emulate(gcm, rcm, rcps, rcp_std, topology, vars, path_predictors, type, t_std, t_train, t_test, 
            perfect = False, predictand = "tas", scale = True, bias_correction = False):
    
    time = t_test
    print(time)

    ## Create the list period strings for the paths to the files
    t_std = time_comb(t_std)
    print(t_std)
    t_tests = time_comb(t_test)
    print(t_tests)

    ## Load predictor base data (use in traning for standardized) for PP-E
    if type == 'PP-E':
        training_dataset = 'upscaledrcm'
        path_base = [xr.open_dataset(f'./data/predictors/{training_dataset}/x_cnrm-ald63_{rcp_std}_{t}.nc') for t in t_std]
        base = xr.concat(path_base, dim='time')
    ## Load predictor base data (use in traning for standardized) for MOS-E            
    elif type == 'MOS-E':
        training_dataset = 'gcm'
        path_base = [xr.open_dataset(f'./data/predictors/{training_dataset}/x_cnrm_{rcp_std}_{t}.nc') for t in t_std]
        base = xr.concat(path_base, dim='time')
    
    for rcp_train in rcps: # rcp use in train
        for rcp_test in rcps: # rcp use in test

            files = [xr.open_dataset(f'./data/predictand/tas/{predictand}_{gcm}-{rcm}_{rcp_test}_{t}.nc') for t in t_tests]
            y = xr.concat(files, dim="time")

            if type == 'PP-E' and perfect is True:
                training_dataset = 'upscaledrcm'
                ### Load predictor data (.nc)
                files_x = [xr.open_dataset(f'{path_predictors}{training_dataset}/x_{gcm}-{rcm}_{rcp_test}_{t}.nc') for t in t_tests]
                x = xr.concat(files_x, dim='time')

            else:
                training_dataset = 'gcm'
                ### Load predictor data (.nc)
                files_x = [xr.open_dataset(f'{path_predictors}{training_dataset}/x_{gcm}_{rcp_test}_{t}.nc') for t in t_tests]
                x = xr.concat(files_x, dim='time')

            if vars is not None:
                x = x[vars]
                base = base[vars]
        
            modelPath = f'./models/{predictand}/{topology}-{predictand}-{type}-{gcm}-{rcm}-{rcp_train}_{t_train[0]}-{t_train[1]}.h5'
            
            ## Bias correction?..
            if bias_correction is True:
                print('bias correction...')
                # load base
                training_dataset = 'gcm'
                files_base = [xr.open_dataset(f'{path_predictors}{training_dataset}/x_{gcm}_{rcp_test}_{t}.nc') for t in t_tests]
                base_gcm = xr.concat(files_base, dim='time')
                # load ref
                training_dataset = 'upscaledrcm'
                files_ref = [xr.open_dataset(f'{path_predictors}{training_dataset}/x_{gcm}-{rcm}_{rcp_test}_{t}.nc') for t in t_tests]
                base_ref = xr.concat(files_ref, dim='time')

                if vars is not None:
                    base_gcm = base_gcm[vars]
                    base_ref = base_ref[vars]

                x = scaleGrid(x, base = base_gcm, ref = base_ref, type = 'center', timeFrame = 'monthly', spatialFrame = 'gridbox')

            ## Scaling..
            if scale is True:
                print('scaling...')
                x = scaleGrid(x, base = base, type = 'standardize', spatialFrame = 'gridbox')

            ## Loading the cnn model...
            if predictand == 'tas' or predictand == 'tasmin' or predictand == 'tasmax':
                model = tf.keras.models.load_model(modelPath)
                description = {'description': 'air surface temperature (ºC)'}

            ## Converting xarray to a numpy array and predict on the test set
            x_array = x.to_stacked_array("var", sample_dims = ["lon", "lat", "time"]).values
            pred = model.predict(x_array)

            ## Reshaping the prediction to a latitude-longitude grid
            mask = xr.open_dataset('./data/land_sea_mask/lsm_ald63.nc')
            if topology == 'deepesd':
                mask.sftlf.values[mask.sftlf.values == 0] = np.nan
                mask_Onedim = mask.sftlf.values.reshape((np.prod(mask.sftlf.shape)))
                ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 1]
                pred = reshapeToMap(grid = pred, ntime = x.dims['time'], nlat = mask.dims['lat'], nlon = mask.dims['lon'], indLand = ind)

            ## Create a xarray dataset with the prediction
            pred = xr.Dataset(
            data_vars = {predictand: (['time', 'lat', 'lon'], pred)},
            coords = {'lon': mask.lon.values, 
                      'lat': mask.lat.values, 
                      'time': x.time.values},
            attrs = description
            )

            ## Save the prediction to a netcdf file
            outputFileName = f'./pred/{predictand}/{type}-BC_{predictand}_{topology}_alp12_{gcm}-{rcm}_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{time[0]}-{time[1]}.nc' 
            pred.to_netcdf(outputFileName)
            print(outputFileName)





