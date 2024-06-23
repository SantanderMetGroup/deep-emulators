from buildEmulator import buildEmulator

# Variables to be used in the emulator

vars = ['zg500', 'zg700',
        'hus500', 'hus700', 'hus850',
        'ua500', 'ua700', 'ua850',
        'va500', 'va700', 'va850',
        'ta500', 'ta700', 'ta850']

predictand = 'tas'
gcm = 'cnrm'
rcm = 'ald63'
rcp = 'rcp26'
predictorsUpscaledPath = '../../data/predictors/upscaledrcm/'
predictorsGCMPath = '../../data/predictors/gcm/'
predictandRootPath = f'../../data/predictand/{predictand}/'
modelRootPath = '../../m/'
aprox = 'PP-E'
t_train = [2010, 2029]
t_base = [2010, 2019]
topology = 'deepesd'
# Auxiliar funtions to generate the paths of the files

def timeCombinations(time):
    '''
    Create the time combinations strings

    :param time: list, time period [start, end] in years

    :return: list, time combinations strings
    '''
    combinations = []
	# Loop through decades from the start year to the end year
    for start_year in range(time[0], time[1] + 1, 10):
        end_year = min(start_year + 9, time[1])  # Calculate the end year of the decade
        combination = f"{start_year}-{end_year}"  # Construct the combination string
        combinations.append(combination)  # Add the combination to the list
        if end_year == time[1]:
            break
    return combinations

def filePaths(time, path, rcp):
    '''
    Generate the paths of the files to be used in the emulator
    
    :param time: list, time period
    :param path: str, path to the predictors data
    :param rcp: str, rcp scenario
    
    :return: list, paths of the files
    '''

    periods = timeCombinations(time)
    paths = []
    for period in periods:
        if path == '../../data/predictors/upscaledrcm/':
            paths.append(f'{path}x_{gcm}-{rcm}_{rcp}_{period}.nc')
        elif path == '../../data/predictors/gcm/':
            paths.append(f'{path}x_{gcm}_{rcp}_{period}.nc')
        else:
            paths.append(f'{path}{predictand}_{gcm}-{rcm}_{rcp}_{period}.nc')
    return paths

def modelPaths(path, rcp, time, predictand, aprox,gcm, rcm):
    '''
    Generate the path of the model
    
    :param modelRootPath: str, path to the model
    :param rcp: str, rcp scenario
    :param t_train: list, time period of the training data
    
    :return: str, path of the model
    '''
    
    return f'{path}deepesd-{predictand}-{aprox}-{gcm}-{rcm}-{rcp}_{time[0]}-{time[1]}.h5'
    
predictorsPath = filePaths(t_train, predictorsUpscaledPath, 'rcp26')
basePath = filePaths(t_base, predictorsGCMPath, 'rcp45')
predictandPath = filePaths(t_train, predictandRootPath, 'rcp26')
modelPath = modelPaths(modelRootPath, 'rcp26', t_train, predictand, aprox, gcm, rcm)
maskPath = '../../data/land_sea_mask/lsm_ald63.nc'
buildEmulator(predictorsPath = predictorsPath,
                basePath = basePath,
                predictandPath = predictandPath,
                modelPath = modelPath,
                maskPath = maskPath,
                topology = topology,
                predictand = predictand,
                vars = vars,
                scale = True)

