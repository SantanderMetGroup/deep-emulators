def accumulated_saliency_map(grid, model, months, total_num_sites, outputFileName, percentage = True, samples = None, baseline=None, num_steps=50):

    ## Converting xarray to a numpy array and predict on the test set
    grid = xr.merge([grid.sel(time = grid['time.month'] == month) for month in months])
    grid_array = grid.to_stacked_array("var", sample_dims = ["lon", "lat", "time"]).values

    ## Comuting integrated gradients
    asm_site = []
    for site in range(5):
        ig = get_integrated_gradients(img_input = grid_array,
                                      top_pred_idx = site,
                                      model = model,
                                      baseline = baseline,
                                      num_steps = num_steps)
        vars_names = list(grid.keys())
        new_data_vars = {}
        for ind_vn in range(len(vars_names)):
            new_data_vars.update({vars_names[ind_vn]: (['time', 'lat', 'lon'], ig[:,:,:,ind_vn])})
            ig_dataset = xr.Dataset(data_vars = new_data_vars,
                                    coords = grid.coords,
                                    attrs = {'description': 'integrated gradients'})

        print(str(site) + '/' + str(total_num_sites - 1))
        if percentage is True:
            print('computing percentage...')
            ig_dataset = xr.concat([np.abs(ig_dataset.sel(time = ig_dataset.time[day])) / np.abs(ig_dataset.sel(time = ig_dataset.time[day])).to_stacked_array("var", sample_dims = ["lon", "lat"]).sum() for day in range(len(ig_dataset.time))], dim = 'time') # , compat = 'override')
            print('done')
        ig_dataset_mean = ig_dataset.mean('time')
        asm_site.append(ig_dataset_mean)
    asm = xr.concat(asm_site, dim = 'site').mean('site')
    asm.to_netcdf(outputFileName)



def integrated_gradients(grid, model, site, outputFileName, filter, percentage = True, samples = None, baseline=None, num_steps=50):

    ## Converting xarray to a numpy array and predict on the test set
    if samples is not None:
        grid = grid.sel(time = samples)
    grid_array = grid.to_stacked_array("var", sample_dims = ["lon", "lat", "time"]).values

    ## Comuting integrated gradients
    ig = get_gradients_vanilla(img_input = grid_array,
                                  top_pred_idx = site,
                                  model = model)

    vars_names = list(grid.keys())
    new_data_vars = {}
    for ind_vn in range(len(vars_names)):
        new_data_vars.update({vars_names[ind_vn]: (['time', 'lat', 'lon'], ig[:,:,:,ind_vn])})

    ig_dataset = xr.Dataset(
		data_vars = new_data_vars,
	    	coords = grid.coords,
	    	attrs = {'description': 'integrated gradients'})

    print(outputFileName)
    if percentage is True:
        print('computing percentage...')
        ig_x = np.abs(ig_dataset)
        # ig_x = ig_x.where(ig_x > 0.0015, 0)
        ig_dataset = xr.concat([ig_x.sel(time = ig_x.time[day]) / ig_x.sel(time = ig_x.time[day]).to_stacked_array("var", sample_dims = ["lon", "lat"]).sum() for day in range(len(ig_x.time))], dim = 'time') # , compat = 'override')
        ig_dataset = ig_dataset.where(ig_dataset > 0.0015, 0)
        ig_dataset = xr.concat([ig_dataset.sel(time = ig_dataset.time[day]) / ig_dataset.sel(time = ig_dataset.time[day]).to_stacked_array("var", sample_dims = ["lon", "lat"]).sum() for day in range(len(ig_dataset.time))], dim = 'time') # , compat = 'override')
        print('done')
    ig_dataset.to_netcdf(outputFileName)


def get_gradients(img_input, top_pred_idx, model):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        img_input: 4D image tensor
        top_pred_idx: Predicted label for the input image

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        # print(images.shape)
        preds = model(images)
        top_class = preds[:, top_pred_idx]

    grads = tape.gradient(top_class, images)
    return grads


def get_gradients_vanilla(img_input, top_pred_idx, model):
    """Computes Integrated Gradients for a predicted label.

    Args:
        img_input (ndarray): Original image
        top_pred_idx: Predicted label for the input image
        baseline (ndarray): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        Integrated gradients w.r.t input image
    """
    print("qué marcha me llevas")
    
    # 2. Get the gradients
    # print(i)
    # print(img.shape)
    # img = tf.expand_dims(img, axis=0)
    grad = get_gradients(img_input, top_pred_idx=top_pred_idx, model = model)
    grad = tf.convert_to_tensor(grad, dtype=tf.float32)

    return grad
