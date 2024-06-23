from auxiliaryFunctions import applyMask, openFiles, scaleGrid, trainModel
from deepmodel import deepmodel
import numpy as np
import sys

def buildEmulator(predictorsPath, basePath, predictandPath, modelPath, maskPath, topology, predictand,vars, scale = True):
	'''
	Build the emulator

	:param predictorsPath: str, path to the predictors data
	:param basePath: str, path to the base data
	:param predictandPath: str, path to the predictand data
	:param modelPath: str, path to save the model
	:param maskPath: str, path to the mask
	:param topology: str, topology of the model
	:param predictand: str, predictand
	:param scale: boolean, scale the data
	'''

	## Open the predictors data (.nc)
	x = openFiles(predictorsPath)
	
	if vars is not None:
		x = x[vars]
	
	## Scaling..
	if scale:
		print("Scaling data...")
		sys.stdout.flush()

		base = openFiles(basePath)
		x = scaleGrid(x, base = base, type = 'standardize', spatialFrame = 'gridbox')

	## Open the predictand data (.nc)
	y = openFiles(predictandPath)

	## Intersecting the time dimension
	ind_time = np.intersect1d(y.time.values, x.time.values)
	x = x.sel(time = ind_time)
	y = y.sel(time = ind_time)

	## Converting xarray to a numpy array
	x_array = x.to_stacked_array("var", sample_dims = ["lon", "lat", "time"]).values

	## Mask the sea in the deepesd model
	yTrain = applyMask(maskPath, y, x, predictand)
	outputShape = yTrain.shape[1]

	## Define deep model
	model = deepmodel(topology = topology,
					  predictand = predictand,
					  inputShape = x_array.shape[1::],
					  outputShape = outputShape)

	## Train the model
	trainModel(x = x_array, 
			y = yTrain, 
			model = model, 
			modelPath= modelPath, 
			predictand = predictand)