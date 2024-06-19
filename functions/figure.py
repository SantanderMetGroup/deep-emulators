def metricas(grid_obs, grid_prd,var,metric):
    if metric == 'bias':
            grid = bias(grid_obs, grid_prd, var=var)
            grid_z = grid
            levels = np.linspace(-2, 2, 17)
            cmap = 'RdBu_r'
    if metric == 'rmse':
            grid = rmse(grid_obs, grid_prd, var=var)
            grid_z = grid
            levels = np.linspace(0, 3, 17)
            cmap = 'Reds'
    if metric == 'stdRatio':
            grid = stdRatio(grid_obs, grid_prd, var=var)
            grid_z = grid
            levels = np.linspace(-4, 4, 17)
            cmap = 'Reds'
    if metric == 'P02':
            grid = biasPercent(grid_obs, grid_prd, var=var, per=0.02)
            grid_z = grid
            levels = np.linspace(-2, 2, 17)
            cmap = 'RdBu_r'
    if metric == 'P98':
            grid = biasPercent(grid_obs, grid_prd, var=var, per=0.98)
            grid_z = grid
            levels = np.linspace(-2, 2, 17)
            cmap = 'RdBu_r'
    return grid, grid_z, levels, cmap

def rmse(grid_obs, grid_prd, var):
        obs2 = grid_obs
        prd2 = grid_prd
        rmse = np.sqrt(((obs2[var] - prd2[var])**2).mean(dim='time'))
        return rmse

def stdRatio(grid_obs, grid_prd, var):
        obs2 = grid_obs
        prd2 = grid_prd
        stdRatio = (prd2[var].std(dim='time') ) / obs2[var].std(dim='time')
        return stdRatio

def mean(grid_obs, grid_prd, var):
    obs2 = grid_obs
    prd2 = grid_prd
    mean = obs2[var].mean(dim='time') - prd2[var].mean(dim='time')
    return mean

def biasPercent(obs, prd, var, per):
    obs2 = obs
    prd2 = prd
    obs2 = obs2.chunk(dict(time=-1))
    prd2 = prd2.chunk(dict(time=-1))
    obs2_p98 = obs2.quantile(per, 'time', skipna=True)
    prd2_p98 = prd2.quantile(per, 'time', skipna=True)
    bias = (prd2_p98 - obs2_p98) 
    return bias[var]

def bias(obs, prd, var):
    obs2 = obs
    prd2 = prd
    bias_values = np.mean(prd2[var].values, axis = 0) -  np.mean(obs2[var].values, axis = 0)
    template = prd2[var].mean('time')
    template.values = bias_values
    return template


def panel(gcm, rcm, times, t_train, t_test, rcps, path_predictand, predictand, figsize, type, BC = False, perfect=False):

    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(4, 1, figure=fig, hspace=0.2)  # Adjust hspace for spacing between periods

    def create_subplots(gs, fig):
        inner_gs = GridSpecFromSubplotSpec(2, 11, subplot_spec=gs, wspace=0.1, hspace=0.05, width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.2, 1, 1])  # Adjust ratios as needed
        axs = []
        for i in range(2):  # For each row
            row = []
            for j in range(11):  # For each column
                if j % 3 != 2:  # Skip the padding columns
                    ax = fig.add_subplot(inner_gs[i, j], projection=ccrs.PlateCarree())
                    row.append(ax)
                else:
                    row.append(None)  # Append None for the padding columns
            axs.append(row)
        return axs

    # Create 4 sets of subplots for each metric in each period
    period1_axs = create_subplots(outer_gs[0], fig)
    period2_axs = create_subplots(outer_gs[1], fig)
    period3_axs = create_subplots(outer_gs[2], fig)
    period4_axs = create_subplots(outer_gs[3], fig)

    periods_axs = [period1_axs, period2_axs, period3_axs, period4_axs]
    subtitles = ['2010-2029', '2040-2059', '2060-2079', '2080-2099']
    metrics = ['bias', 'rmse', 'P02', 'P98']

    # Add period subtitles to the middle of each row
    for i, subtitle in enumerate(subtitles):
        fig.text(0.10, 0.795 - i * 0.2, subtitle, va='center', ha='center', rotation='vertical', fontsize=12, fontweight='bold')

    # Add titles above each set of columns
    metric_titles = ['Bias', 'RMSE', 'P02', 'P98']
    for i, metric_title in enumerate(metric_titles):
        fig.text(0.21 + i * 0.2, 0.90, metric_title, va='center', ha='center', fontsize=12, fontweight='bold')

    # Add RCP titles above each column set
    rcp_titles = ['RCP 4.5', 'RCP 8.5']
    for i in range(2):  # There are two RCPs
        for j in range(4):  # There are four metrics        
            fig.text(0.12, 0.84 - j * 0.20 - i * 0.084, rcp_titles[i], va='center', ha='center', rotation='vertical', fontsize=10, fontweight='bold')
            fig.text(0.175 + j * 0.20 + i * 0.08, 0.88, rcp_titles[i], va='center', ha='center', fontsize=10, fontweight='bold')

    all_contours = []

    if type == 'MOS-E':
        outputFileName = f'fig/{predictand}/panel_MOS-E_{predictand}_train-{t_train[0]}-{t_train[1]}.pdf'
    elif BC:
        outputFileName = f'fig/{predictand}/panel_PP-E-BC_{predictand}_train-{t_train[0]}-{t_train[1]}.pdf'
    elif BC is False:
        outputFileName = f'fig/{predictand}/panel_PP-E_{predictand}_train-{t_train[0]}-{t_train[1]}.pdf'
    else: 
        raise ValueError('Invalid type specified.')
    
    for k, time in enumerate(times):
        for i, rcp_test in enumerate(rcps):
            for j, rcp_train in enumerate(rcps):
                if type == 'MOS-E':
                    grid_prd = xr.open_dataset(f'./pred/{predictand}/MOS-E_{predictand}_deepesd_alp12_cnrm-ald63_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc').sel(time=slice(f'{time[0]}-01-01', f'{time[1]}-12-31'))
                elif BC:
                    grid_prd = xr.open_dataset(f'./pred/{predictand}/PP-E-BC_{predictand}_deepesd_alp12_cnrm-ald63_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc').sel(time=slice(f'{time[0]}-01-01', f'{time[1]}-12-31'))
                elif not BC:
                    grid_prd = xr.open_dataset(f'./pred/{predictand}/PP-E_{predictand}_deepesd_alp12_cnrm-ald63_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc').sel(time=slice(f'{time[0]}-01-01', f'{time[1]}-12-31'))
                else: 
                    raise ValueError('Invalid type specified.')

                combinations = time_comb(time)
                files = [xr.open_dataset(f'{path_predictand}{predictand}/{predictand}_{gcm}-{rcm}_{rcp_test}_{t}.nc') for t in combinations]
                grid_obs = xr.concat(files, dim='time')

                for m, metric in enumerate(metrics):
                    grid, grid_z, levels, cmap = metricas(grid_obs, grid_prd, var=predictand, metric=metric)
                    ax = periods_axs[k][i][j + m * 3]
                    if ax:  # Only plot if ax is not None (skip padding columns)
                        ax.coastlines()
                        fig_ = ax.contourf(grid.lon, grid.lat, grid_z, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
                        all_contours.append(fig_)
                plt.savefig(outputFileName)

        print('period done')

    # Add colorbars for each metric

    for i, contour in enumerate(all_contours[:4]):
        cbar_ax = fig.add_axes([0.132 + i * 0.2, 0.09, 0.160, 0.014])
        fig.colorbar(contour, orientation='horizontal', cax=cbar_ax)

    plt.savefig(outputFileName)
    print(outputFileName)
    plt.close()

def panel_perfect(gcm, rcm, times, t_train, t_test, rcps, path_predictand, predictand, figsize):

    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(4, 1, figure=fig, hspace=0.2)  # Adjust hspace for spacing between periods

    def create_subplots(gs, fig):
        inner_gs = GridSpecFromSubplotSpec(3, 15, subplot_spec=gs, wspace=0.1, hspace=0.05, width_ratios=[1, 1, 1, 0.2, 1, 1, 1, 0.2, 1, 1, 1, 0.2, 1, 1, 1])  # Adjust ratios as needed
        axs = []
        for i in range(3):  # For each row
            row = []
            for j in range(15):  # For each column
                if j % 4 != 3:  # Skip the padding columns
                    ax = fig.add_subplot(inner_gs[i, j], projection=ccrs.PlateCarree())
                    row.append(ax)
                else:
                    row.append(None)  # Append None for the padding columns
            axs.append(row)
        return axs

    # Create 4 sets of subplots for each metric in each period
    period1_axs = create_subplots(outer_gs[0], fig)
    period2_axs = create_subplots(outer_gs[1], fig)
    period3_axs = create_subplots(outer_gs[2], fig)
    period4_axs = create_subplots(outer_gs[3], fig)

    periods_axs = [period1_axs, period2_axs, period3_axs, period4_axs]
    subtitles = ['2010-2029', '2040-2059', '2060-2079', '2080-2099']
    metrics = ['bias', 'rmse', 'P02', 'P98']

    # Add period subtitles to the middle of each row
    for i, subtitle in enumerate(subtitles):
        fig.text(0.10, 0.795 - i * 0.20, subtitle, va='center', ha='center', rotation='vertical', fontsize=12, fontweight='bold')

    # Add titles above each set of 3 columns
    metric_titles = ['Bias', 'RMSE', 'P02', 'P98']
    for i, metric_title in enumerate(metric_titles):
        fig.text(0.215 + i * 0.20, 0.90, metric_title, va='center', ha='center', fontsize=12, fontweight='bold')

    # Add RCP titles above each column set
    rcp_titles = ['RCP 2.6', 'RCP 4.5', 'RCP 8.5']
    for i in range(3):  # There are three RCPs
        for j in range(4):  # There are four metrics        
            fig.text(0.12, 0.85 - j * 0.20 - i * 0.055, rcp_titles[i], va='center', ha='center', rotation='vertical', fontsize=10, fontweight='bold')
            fig.text(0.155 + j * 0.20 + i * 0.06, 0.88, rcp_titles[i], va='center', ha='center', fontsize=10, fontweight='bold')

    all_contours = []

    outputFileName = f'fig/{predictand}/panel_PP-E-perfect_{predictand}_train-{t_train[0]}-{t_train[1]}.pdf'

    for k, time in enumerate(times):
        for i, rcp_test in enumerate(rcps):
            for j, rcp_train in enumerate(rcps):
                grid_prd = xr.open_dataset(f'./pred/{predictand}/PP-E-perfect_{predictand}_deepesd_alp12_cnrm-ald63_train-{rcp_train}-{t_train[0]}-{t_train[1]}_test-{rcp_test}-{t_test[0]}-{t_test[1]}.nc').sel(time=slice(f'{time[0]}-01-01', f'{time[1]}-12-31'))
                combinations = time_comb(time)
                files = [xr.open_dataset(f'{path_predictand}{predictand}/{predictand}_{gcm}-{rcm}_{rcp_test}_{t}.nc') for t in combinations]
                grid_obs = xr.concat(files, dim='time')

                for m, metric in enumerate(metrics):
                    grid, grid_z, levels, cmap = metricas(grid_obs, grid_prd, var=predictand, metric=metric)
                    ax = periods_axs[k][i][j + m * 4]
                    if ax:  # Only plot if ax is not None (skip padding columns)
                        ax.coastlines()
                        fig_ = ax.contourf(grid.lon, grid.lat, grid_z, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
                        all_contours.append(fig_)
                plt.savefig(outputFileName)
        print('period done')

    # Add colorbars for each metric

    for i, contour in enumerate(all_contours[:4]):
        cbar_ax = fig.add_axes([0.135 + i * 0.20, 0.08, 0.160, 0.014])
        fig.colorbar(contour, orientation='horizontal', cax=cbar_ax)

    plt.savefig(outputFileName)
    print(outputFileName)
    plt.close()


