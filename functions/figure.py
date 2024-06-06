def figure6(training_gcm, topology, figsize = (10,40), predictand = 'tas', outputFileName = None):
    levels = np.linspace(-2, 2, 17)
    cmap = 'RdBu_r'
    fig, ax = plt.subplots(6, 2, figsize = figsize, subplot_kw = {'projection': ccrs.PlateCarree()})
    set1 = set(['noresm', 'mpi', 'cnrm'])
    set2 = set([training_gcm])
    gcms = list(set1.difference(set2))
    for i in range(6):
        if i == 0 or i == 2 or i == 4:
            type = 'PP-E'
        if i == 1 or i == 3 or i == 5:
            type = 'MOS-E'
        for j in range(2):
            if j == 0:
                grid_obs = xr.open_dataset('../data/predictand/tas_' + gcms[0] + '-ald63_rcp85_2041-2050_v2.nc')    
                grid_prd = xr.open_dataset('../predictions/hard-trans/' + type + '_' + topology + '_' + predictand + '_alp12_' + 'PREDGCM:' + gcms[0] + '_TRAGCM:' + training_gcm + '_RCM:ald63_rcp85_2041-2050.nc')
            if j == 1:
                grid_obs = xr.open_dataset('../data/predictand/tas_' + gcms[1] + '-ald63_rcp85_2041-2050_v2.nc')   
                grid_prd = xr.open_dataset('../predictions/hard-trans/' + type + '_' + topology + '_' + predictand + '_alp12_' + 'PREDGCM:' + gcms[1] + '_TRAGCM:' + training_gcm + '_RCM:ald63_rcp85_2041-2050.nc')
            if i == 0 or i == 1:
                grid = bias(grid_obs, grid_prd, var = 'tas')
                grid_z = grid
            if i == 2 or i == 3:
                grid = bias(grid_obs, grid_prd, period = 'winter', var = 'tas')
                grid_z = grid
            if i == 4 or i == 5:
                grid = bias(grid_obs, grid_prd, period = 'summer', var = 'tas')
                grid_z = grid
            ax[i,j].coastlines()
            fig_ = ax[i,j].contourf(grid.lon,
                                    grid.lat,
                                    grid_z,
                                    levels = levels,
                                    cmap = cmap,
                                    extend = 'both',
                                    transform = ccrs.PlateCarree()
                                    )
            fig.colorbar(fig_, ax = ax[i,j])
    if outputFileName is None:
    	plt.show()
    else:
    	plt.savefig(outputFileName, bbox_inches = 'tight')
    plt.close()



def figure4(groundtruth, topology, figsize, outputFileName = None, predictand = "tas"):
    plt.close()
    if predictand == 'pr':
        metrics = ['biasR01', 'biasSDII', 'biasP98Wet', 'rmse']
    elif predictand == 'tas':
        metrics = ['bias', 'biasWinter', 'biasSummer', 'rmse']

    fig, ax = plt.subplots(4, 5*len(groundtruth), figsize = figsize, subplot_kw = {'projection': ccrs.PlateCarree()})
    j = -1
    for gcm in groundtruth:
        grid_test = xr.open_dataset('../data/predictand/' + predictand + '_' + gcm + '-ald63_rcp85_2041-2050.nc')
        grid_base_h = xr.open_dataset('../data/predictand/' + predictand + '_' + gcm + '-ald63_historical_1996-2005.nc')
        grid_base_f = xr.open_dataset('../data/predictand/' + predictand + '_' + gcm + '-ald63_rcp85_2090-2099.nc')
        grid_base = xr.concat([grid_base_h,grid_base_f], dim = 'time')
        for case in range(5):
            j = j + 1
            if case == 0:
                pred_path = '../predictions/soft-trans/MOS-E_' + topology + '_' + predictand + '_alp12_' + gcm + '-ald63_train.nc'
                grid_obs = grid_base
            if case == 1:
                pred_path = '../predictions/soft-trans/PP-E-perfect_' + topology + '_' + predictand + '_alp12_' + gcm + '-ald63_train.nc'
                grid_obs = grid_base
            if case == 2:
                pred_path = '../predictions/soft-trans/MOS-E_' + topology + '_' + predictand + '_alp12_' + gcm + '-ald63_rcp85_2041-2050.nc'
                #pred_path = '../predictions/soft-trans/PP-E-perfect_' + topology + '_' + predictand + '_alp12_' + gcm + '-ald63_rcp85_2041-2050.nc'
                grid_obs = grid_test
            if case == 3:
                pred_path = '../predictions/soft-trans/PP-E_' + topology + '_' + predictand + '_alp12_' + gcm + '-ald63_rcp85_2041-2050.nc'
                grid_obs = grid_test
            if case == 4:
                pred_path = '../predictions/soft-trans/PP-E-bc_' + topology + '_' + predictand + '_alp12_' + gcm + '-ald63_rcp85_2041-2050.nc'
                grid_obs = grid_test
            grid_prd = xr.open_dataset(pred_path)
            for ind_metric in range(len(metrics)):
                levels = np.linspace(-0.4, 0.4, 17)
                if metrics[ind_metric] == 'biasR01':
                    print("ahora R01")
                    grid = biasR01(grid_obs, grid_prd, var = 'pr').pr
                    grid_z = grid
                    cmap = 'BrBG'
                if metrics[ind_metric] == 'biasSDII':
                    print("ahora SDII")
                    grid = biasSDII(grid_obs, grid_prd, var = 'pr').pr
                    grid_z = grid
                    cmap = 'BrBG'
                if metrics[ind_metric] == 'biasP98Wet':
                    print("ahora P98Wet")
                    grid = biasP98Wet(grid_obs, grid_prd, var = 'pr').pr
                    grid_z = grid
                    cmap = 'BrBG'
                if metrics[ind_metric] + '_' + predictand == 'rmse_pr':
                    print("ahora rmse")
                    ind_time = np.intersect1d(grid_obs.time.values, grid_prd.time.values)
                    grid_obs_2 = grid_obs.sel(time = ind_time)
                    grid_prd_2 = grid_prd.sel(time = ind_time)
                    grid = rmse(grid_obs_2, grid_prd_2, var = 'pr')
                    grid_z = grid
                    levels = np.linspace(2, 10, 17)
                    cmap = 'Blues'
                if metrics[ind_metric] + '_' + predictand == 'corr_pr':
                    print("ahora corr")
                    ind_time = np.intersect1d(grid_obs.time.values, grid_prd.time.values)
                    grid_obs_2 = grid_obs.sel(time = ind_time)
                    grid_prd_2 = grid_prd.sel(time = ind_time)
                    grid_obs_2_year = grid_obs_2.groupby('time.year').sum('time')
                    grid_prd_2_year = grid_prd_2.groupby('time.year').sum('time')
                    grid = xr.corr(grid_obs_2_year.pr, grid_prd_2_year.pr, dim = 'year')
                    grid_z = grid
                    levels = np.linspace(0.6, 1, 17)
                    cmap = 'Blues'
                if metrics[ind_metric] == 'bias':
                    grid = bias(grid_obs, grid_prd, var = 'tas')
                    grid_z = grid
                    levels = np.linspace(-2, 2, 17)
                    cmap = 'RdBu_r'
                if metrics[ind_metric] == 'biasWinter':
                    grid = bias(grid_obs, grid_prd, var = 'tas', period = 'winter')
                    grid_z = grid
                    levels = np.linspace(-2, 2, 17)
                    cmap = 'RdBu_r'
                if metrics[ind_metric] == 'biasSummer':
                    grid = bias(grid_obs, grid_prd, var = 'tas', period = 'summer')
                    grid_z = grid
                    levels = np.linspace(-2, 2, 17)
                    cmap = 'RdBu_r'
                if metrics[ind_metric] + '_' + predictand == 'rmse_tas':
                    ind_time = np.intersect1d(grid_obs.time.values, grid_prd.time.values)
                    grid_obs_2 = grid_obs.sel(time = ind_time)
                    grid_prd_2 = grid_prd.sel(time = ind_time)
                    grid = rmse(grid_obs_2, grid_prd_2)
                    grid_z = grid
                    levels = np.linspace(0, 4, 17)
                    cmap = 'Reds'
                if metrics[ind_metric] + '_' + predictand == 'corr_tas':
                    ind_time = np.intersect1d(grid_obs.time.values, grid_prd.time.values)
                    grid_obs_2 = grid_obs.sel(time = ind_time)
                    grid_prd_2 = grid_prd.sel(time = ind_time)
                    grid_obs_2_year = grid_obs_2.groupby('time.year').mean('time')
                    grid_prd_2_year = grid_prd_2.groupby('time.year').mean('time')
                    grid = xr.corr(grid_obs_2_year.tas, grid_prd_2_year.tas, dim = 'year')
                    grid_z = grid
                    levels = np.linspace(0.6, 1, 17)
                    cmap = 'Reds'
                ax[ind_metric,j].coastlines()
                fig_ = ax[ind_metric,j].contourf(grid.lon,
                                                 grid.lat,
                                                 grid_z,
                                                 levels = levels,
                                                 cmap = cmap,
                                                 extend = 'both',
                                                 transform = ccrs.PlateCarree()
                                                )
                fig.colorbar(fig_, ax = ax[ind_metric,j])
    if outputFileName is None:
    	plt.show()
    else:
    	plt.savefig(outputFileName, bbox_inches = 'tight')
    plt.close()




