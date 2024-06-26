from auxiliaryFunctions import applyMask, openFiles, scaleGrid, biasCorrection,createDataset
import numpy as np
import sys
import tensorflow as tf

def emulate(predictionPath, predictorsPath, basePath, modelPath, maskPath, description,
            predictand, vars, scale = True, BC = False, baseGCMPath = None, baseRefPath = None):
    
    '''
    Emulate the predictand using the trained model 

    :param predictorsPath: str, path to the predictors data
    :param basePath: str, path to the base data
    :param predictandPath: str, path to the predictand data
    :param modelPath: str, path to the trained model
    :param maskPath: str, path to the mask
    :param topology: str, topology of the model
    :param predictand: str, predictand
    :param scale: boolean, scale the data
    :param biasCorrection: boolean, apply bias correction

    :return: None
    '''
    x = openFiles(predictorsPath)

    if vars is not None:
        x = x[vars]
                
    ## Bias correction?..
    if BC:
        print('bias correction...')
        sys.stdout.flush()

        x = biasCorrection(x, baseGCMPath, baseRefPath, vars)

    if scale:
        print("Scaling data...")
        sys.stdout.flush()

        base = openFiles(basePath)
        if vars is not None:
            base = base[vars]
        x = scaleGrid(x, base = base, type = 'standardize', spatialFrame = 'gridbox')
        
    model = tf.keras.models.load_model(modelPath) 

    ## Converting xarray to a numpy array and predict on the test set
    x_array = x.to_stacked_array("var", sample_dims = ["lon", "lat", "time"]).values
    pred = model.predict(x_array)

    pred = applyMask(maskPath, pred, x, predictand, 'emulate')

    ## Create a xarray dataset with the prediction
    pred = createDataset(pred, x, description, predictand, maskPath)

    ## Save the prediction to a netcdf file
    pred.to_netcdf(predictionPath)
    print(predictionPath)
    sys.stdout.flush()





