# ######################################### SALIENCY ############################################################
def return_site_neuron(i, j, maskPath = './data/land_sea_mask/lsm_ald63.nc'):
    mask = xr.open_dataset(maskPath)
    mask.sftlf.values[i,j] = 2
    mask_Onedim = mask.sftlf.values.reshape((np.prod(mask.sftlf.shape)))
    ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] != 0]
    ind_site = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 2]
    site = ind.index(ind_site[0])
    return site

def saliency_maps_ppe(vars, gcms, rcm, month, types, epsilon, type_plot, figsize, by_column, by_row, site, inputDir, outputDir, topology, predictand = "tas" , period = 'train'):
    fig, ax = plt.subplots(by_row, by_column, figsize = figsize, subplot_kw = {'projection': ccrs.PlateCarree()})
    j = -1
    mask = xr.open_dataset('./data/land_sea_mask/lsm_ald63.nc')
    site_i = site[0]
    site_j = site[1]
    site = return_site_neuron(i = site_i, j = site_j)
    for ind_emul_type in range(len(types)):
        for gcm in gcms:
            j = j + 1
            print("hello")
            type = types[ind_emul_type]
            grid_ = xr.open_dataset(inputDir + f'/SALIENCY_{topology}_{predictand}_alp12_{gcm}-{rcm}_{type}_site{site}_period:{period}_month:{month[0]}.nc')
            for i in range(len(vars)):
                var = vars[i]
                print(f'site:{site}type:{type}_gcm:{gcm}_var:{var}')
                cmap = 'Reds'
                # Compute saliency
                grid = np.log(grid_[var].mean('time')*1000 + epsilon)
                range_scale = np.linspace(0, np.log(0.01*1000 + epsilon), 17)
                value = str(np.round(np.sum(grid_[var].mean('time'))*100,decimals=2).values)
                # Plotting
                ax[i,j].coastlines()
                ax[i,j].set_title(var + ' (' + value + ')')
                if type_plot == 'contour':
                    fig_ = ax[i,j].contourf(grid.lon,
                                            grid.lat,
                                            grid,
                                            levels = range_scale,
                                            cmap = cmap,
                                            extend = 'both',
                                            transform = ccrs.PlateCarree())
                ax[i,j].scatter(mask.lon.values[site_i, site_j],
                                mask.lat.values[site_i, site_j],
                                s = 30, color = "black")
                fig.colorbar(fig_, ax = ax[i,j])
    plt.savefig(outputDir + f'/SALIENCY_{predictand}_site:{site}_month:{month}.pdf', bbox_inches = 'tight')
    plt.close()

def compute_saliency_ppe(sites, types, gcms, rcm, rcp, predictand, months, outputDir, vars, topology):
    for inds_site in range(len(sites)):
        site_i = sites[inds_site][0]
        site_j = sites[inds_site][1]
        site = return_site_neuron(i = site_i, j = site_j)
        print(str(site))
        for type in types:
            for gcm in gcms:
                xh = xr.open_dataset(f'./data/predictors/upscaledrcm/x_{gcm}-{rcm}_historical_1996-2005.nc')
                xrcp = xr.open_dataset(f'./data/predictors/upscaledrcm/x_{gcm}-{rcm}_{rcp}_2090-2099.nc')

                base = xr.concat([xh,xrcp], dim = 'time')[vars]
                if type == 'train':
                    x = base
                elif type == 'test':
                    x = xr.open_dataset(f'./data/predictors/upscaledrcm/x_{gcm}-{rcm}_{rcp}_2041-2050.nc')[vars]
                elif type == 'gcm':
                    x = xr.open_dataset(f'./data/predictors/gcm/x_{gcm}_{rcp}_2041-2050.nc')[vars]
                elif type == 'gcm-bc':
                    base_h   = xr.open_dataset(f'./data/predictors/gcm/x_{gcm}_historical_1996-2005.nc')
                    base_rcp  = xr.open_dataset(f'./data/predictors/gcm/x_{gcm}_{rcp}_2090-2099.nc')
                    base_gcm = xr.concat([base_h,base_rcp], dim = 'time')[vars]
                    x = xr.open_dataset(f'./data/predictors/gcm/x_{gcm}_{rcp}_2041-2050.nc')[vars]
                    print('bias correction...')
                    x = scaleGrid(x, base = base_gcm, ref = base, type = 'center', timeFrame = 'monthly', spatialFrame = 'gridbox')
                x = scaleGrid(x, base = base, type = 'standardize', spatialFrame = 'gridbox')
                ## Loading the cnn model...
                model = tf.keras.models.load_model(f'./models/{predictand}/{topology}-{predictand}-{gcm}-{rcm}-{rcp}-{type}.h5')
                ## Computing gradients...
                period=type
                integrated_gradients(grid = x,
                                     model = model,
                                     site = site,
                                     baseline = None,
                                     filter = 0.0015,
                                     num_steps = 30,
                                     percentage = True,
                                     outputFileName = f'{outputDir}/SALIENCY_{topology}_tas_alp12_{gcm}-ald63_PP-E_site{site}_period:{period}_month:year.nc')
                grid = xr.open_dataset(f'{outputDir}/SALIENCY_{topology}_tas_alp12_{gcm}-ald63_PP-E_site{site}_period:{period}_month:year.nc')
                for month in months:
                    if month != 'year':
                        grid2 = grid.sel(time = grid['time.month'] == int(month))
                        outputFileName = f'{outputDir}/SALIENCY_{topology}_tas_alp12_{gcm}-ald63_PP-E_site{site}_period:{period}_month:{month}.nc'
                        grid2.to_netcdf(outputFileName)

def compute_saliency(sites, emul_types, gcms, rcm, rcp, months, path_predictors, outputDir, vars, topology, period = 'train', predictand = 'tas'):
    for inds_site in range(len(sites)):
        site_i = sites[inds_site][0]
        site_j = sites[inds_site][1]
        site = return_site_neuron(i = site_i, j = site_j)
        print(str(site))
        for type in emul_types:
            for gcm in gcms:
                if type == 'PP-E':
                    if period == 'train':
                        xh = xr.open_dataset(f'{path_predictors}upscaledrcm/x_{gcm}-{rcm}_historical_1996-2005.nc')
                        xrcp = xr.open_dataset(f'{path_predictors}upscaledrcm/x_{gcm}-{rcm}_{rcp}_2090-2099.nc')
                        x = xr.concat([xh,xrcp], dim = 'time')[vars]
                    elif period == 'test':
                        x = xr.open_dataset(f'{path_predictors}upscaledrcm/x_{gcm}-{rcm}_{rcp}_2041-2050.nc')
                elif type == 'MOS-E':
                    if period == 'train':
                        xh = xr.open_dataset(f'{path_predictors}gcm/x_{gcm}_historical_1996-2005.nc')
                        xrcp = xr.open_dataset(f'{path_predictors}gcm/x_{gcm}_{rcp}_2090-2099.nc')
                        x = xr.concat([xh,xrcp], dim = 'time')[vars]
                    elif period == 'test':
                        xh = xr.open_dataset(f'./data/gcm/x_{gcm}_{rcp}_2041-2050.nc')
                x = scaleGrid(x, base = x, type = 'standardize', spatialFrame = 'gridbox')
                ## Loading the cnn model...
                if predictand == 'tas':
                    model = tf.keras.models.load_model(f'./models/{predictand}/{topology}-{predictand}-{gcm}-{rcm}-{rcp}-{type}.h5')
                elif predictand == 'pr':
                    model = tf.keras.models.load_model(f'./models/{predictand}/{topology}-{predictand}-{gcm}-{rcm}-{rcp}-{type}.h5',
                                                       custom_objects = {'bernoulliGamma': bernoulliGamma})
                ## Computing gradients...
                integrated_gradients(grid = x,
                                     model = model,
                                     site = site,
                                     baseline = None,
                                     filter = 0.0015,
                                     num_steps = 30,
                                     percentage = True,
                                     outputFileName = f'{outputDir}/SALIENCY_{topology}_{predictand}_alp12_{gcm}-{rcm}-{rcp}_{type}_site{site}_period:{period}_month:year.nc')
                grid = xr.open_dataset(f'{outputDir}/SALIENCY_{topology}_{predictand}_alp12_{gcm}-{rcm}-{rcp}_{type}_site{site}_period:{period}_month:year.nc')
                for month in months:
                    if month != 'year':
                        grid2 = grid.sel(time = grid['time.month'] == int(month))
                        outputFileName = f'{outputDir}/SALIENCY_{topology}_{predictand}_alp12_{gcm}-{rcm}-{rcp}_{type}_site{site}_period:{period}_month:{month}.nc'
                        grid2.to_netcdf(outputFileName)
