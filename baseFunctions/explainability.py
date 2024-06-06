### Installing some libraries not contained in udocker gonzabad
# python -m pip install xarray
# pip install netcdf4
# pip install rpy2

### Loading libraries
import os
import xarray as xr
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from scipy import stats
exec(open("../baseFunctions/auxiliaryFunctions.py").read())


# ### Select sites
# # valle del poo [i=52,j=75] alpes-italia(i=52, j=53) alpes-francia(i=58, j=44), Mont-Blanc(i=60, j=54)
# # cerdeña (i=8, j=63) polonia (i=90, j=100) francia (i=90, j=20)
# sites = [52,75],[52,53],[58,44],[8,63],[90,100],[90,20]
# ### Parameters
# vars = ['zg500', 'zg700',
#         'hus500', 'hus700', 'hus850',
#         'ua500', 'ua700', 'ua850',
#         'va500', 'va700', 'va850',
#         'ta500', 'ta700', 'ta850']
# gcms = ['cnrm']
# rcm = 'ald63'
# topology = 'deepesd'
# predictand = 'tas'
# emul_types = ['PP-E', 'MOS-E']
# vars_to_plot = ['hus700', 'hus850', 'ua850', 'va850', 'ta700', 'ta850']
# months = ['8']


# ######################################### SALIENCY ############################################################
def return_site_neuron(i, j, maskPath = '../data/land_sea_mask/lsm_ald63.nc'):
    mask = xr.open_dataset(maskPath)
    mask.sftlf.values[i,j] = 2
    mask_Onedim = mask.sftlf.values.reshape((np.prod(mask.sftlf.shape)))
    ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] != 0]
    ind_site = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 2]
    site = ind.index(ind_site[0])
    return site
# ######################################### SALIENCY ############################################################
def saliency_maps(vars, gcms, month, types, epsilon, type_plot, figsize, by_column, by_row, site, inputDir, outputDir, predictand = "tas"):
    print('hi')
    fig, ax = plt.subplots(by_row, by_column, figsize = figsize, subplot_kw = {'projection': ccrs.PlateCarree()})
    j = -1
    mask = xr.open_dataset('../data/land_sea_mask/lsm_ald63.nc')
    site_i = site[0]
    site_j = site[1]
    site = return_site_neuron(i = site_i, j = site_j)
    for ind_emul_type in range(len(types)):
        for gcm in gcms:
            j = j + 1
            print("hello")
            type = types[ind_emul_type]
            grid_ = xr.open_dataset(inputDir + '/SALIENCY_' + topology + '_tas_alp12_' + gcm + '-ald63_PP-E_site' + str(site) + '_period:'+ type + '_month:' + str(month) + '.nc')
            for i in range(len(vars)):
                var = vars[i]
                print('site:' + str(site) + 'type:' + type + '_gcm:' + gcm + '_var:' + var)
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
    plt.savefig(outputDir + '/SALIENCY_'+ predictand + '_site:' + str(site) + '_month:' + str(month) + '.pdf', bbox_inches = 'tight')
    plt.close()
# ######################################### SALIENCY ############################################################
### Installing some libraries not contained in udocker gonzabad
# python -m pip install xarray
# pip install netcdf4
# pip install rpy2

### Loading libraries
import os
import xarray as xr
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from scipy import stats
exec(open("../baseFunctions/auxiliaryFunctions.py").read())


# ### Select sites
# # valle del poo [i=52,j=75] alpes-italia(i=52, j=53) alpes-francia(i=58, j=44), Mont-Blanc(i=60, j=54)
# # cerdeña (i=8, j=63) polonia (i=90, j=100) francia (i=90, j=20)
# sites = [52,75],[52,53],[58,44],[8,63],[90,100],[90,20]
# ### Parameters
# vars = ['zg500', 'zg700',
#         'hus500', 'hus700', 'hus850',
#         'ua500', 'ua700', 'ua850',
#         'va500', 'va700', 'va850',
#         'ta500', 'ta700', 'ta850']
# gcms = ['cnrm']
# rcm = 'ald63'
# topology = 'deepesd'
# predictand = 'tas'
# emul_types = ['PP-E', 'MOS-E']
# vars_to_plot = ['hus700', 'hus850', 'ua850', 'va850', 'ta700', 'ta850']
# months = ['8']


# ######################################### SALIENCY ############################################################
def return_site_neuron(i, j, maskPath = '../data/land_sea_mask/lsm_ald63.nc'):
    mask = xr.open_dataset(maskPath)
    mask.sftlf.values[i,j] = 2
    mask_Onedim = mask.sftlf.values.reshape((np.prod(mask.sftlf.shape)))
    ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] != 0]
    ind_site = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 2]
    site = ind.index(ind_site[0])
    return site
# ######################################### SALIENCY ############################################################
def saliency_maps(vars, gcms, month, types, epsilon, type_plot, figsize, by_column, by_row, site, inputDir, outputDir, predictand = "tas"):
    fig, ax = plt.subplots(by_row, by_column, figsize = figsize, subplot_kw = {'projection': ccrs.PlateCarree()})
    j = -1
    mask = xr.open_dataset('../data/land_sea_mask/lsm_ald63.nc')
    site_i = site[0]
    site_j = site[1]
    site = return_site_neuron(i = site_i, j = site_j)
    for type in types:
    	for gcm in gcms:
            j = j + 1
            grid_ = xr.open_dataset(inputDir + '/SALIENCY_' + topology + '_tas_alp12_' + gcm + '-ald63_' + type + '_site' + str(site) + '_month:' + str(month) + '.nc')
            for i in range(len(vars)):
                var = vars[i]
                print('site:' + str(site) + 'type:' + type + '_gcm:' + gcm + '_var:' + var)
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
    plt.show()
    plt.savefig(outputDir + '/SALIENCY_'+ predictand + '_site:' + str(site) + '_month:' + str(month) + '.pdf', bbox_inches = 'tight')
    plt.close()

# ######################################### SALIENCY ############################################################

def compute_saliency(sites, types, gcm, rcm, months, vars, outputDir, predictand = 'tas'):
    for inds_site in range(len(sites)):
        site_i = sites[inds_site][0]
        site_j = sites[inds_site][1]
        site = return_site_neuron(i = site_i, j = site_j)
        print(str(site))
        for type in types:
            if type == 'PP-E':
                xh = xr.open_dataset('../data/predictors/upscaledrcm/x_' + gcm + '-' + rcm + '_historical_1996-2005.nc')
                x85 = xr.open_dataset('../data/predictors/upscaledrcm/x_' + gcm + '-' + rcm + '_rcp85_2090-2099.nc')
                x = xr.concat([xh,x85], dim = 'time')[vars]
            if type == 'MOS-E':
                xh = xr.open_dataset('../data/predictors/gcm/x_' + gcm + '_historical_1996-2005.nc')
                x85 = xr.open_dataset('../data/predictors/gcm/x_' + gcm + '_rcp85_2090-2099.nc')
                x = xr.concat([xh,x85], dim = 'time')[vars]
            x = scaleGrid(x, base = x, type = 'standardize', spatialFrame = 'gridbox')
            ## Loading the cnn model...
            if predictand == 'tas':
                model = tf.keras.models.load_model('../models/' + topology + '-' + predictand + '-' + gcm + '-' + rcm + '-' + type + '.h5')
            elif predictand == 'pr':
                model = tf.keras.models.load_model('../models/' + topology + '-' + predictand + '-' + gcm + '-' + rcm + '-' + type + '.h5',
                                               custom_objects = {'bernoulliGamma': bernoulliGamma})
            ## Computing gradients...
            integrated_gradients(grid = x,
                                 model = model,
                                 site = site,
                                 baseline = None,
                                 filter = 0.0015,
                                 num_steps = 30,
                                 percentage = True,
                                 outputFileName = outputDir + '/SALIENCY_' + topology + '_' + predictand + '_alp12_' + gcm + '-ald63_' + type + '_site' + str(site) + '_month:year.nc')
            grid = xr.open_dataset(outputDir + '/SALIENCY_' + topology + '_' + predictand + '_alp12_' + gcm + '-ald63_' + type + '_site' + str(site) + '_month:year.nc')
            for month in months:
                if month != 'year':
                    grid2 = grid.sel(time = grid['time.month'] == int(month))
                    outputFileName = outputDir + '/SALIENCY_' + topology + '_' + predictand + '_alp12_' + gcm + '-ald63_' + type + '_site' + str(site) + '_month:' + str(month) + '.nc'
                    grid2.to_netcdf(outputFileName)


