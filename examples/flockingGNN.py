# 2020/01/01~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
# Kate Tolstaya, eig@seas.upenn.edu

# Learn decentralized controllers for flocking. There is a team of robots that
# start flying at random velocities and we want them to coordinate so that they
# can fly together while avoiding collisions. We learn a decentralized 
# controller by using imitation learning.

# In this simulation, the number of agents is fixed for training, but can be
# set to a different number for testing.

# Outputs:
# - Text file with all the hyperparameters selected for the run and the 
#   corresponding results (hyperparameters.txt)
# - Pickle file with the random seeds of both torch and numpy for accurate
#   reproduction of results (randomSeedUsed.pkl)
# - The parameters of the trained models, for both the Best and the Last
#   instance of each model (savedModels/)
# - The figures of loss and evaluation through the training iterations for
#   each model (figs/ and trainVars/)
# - Videos for some of the trajectories in the dataset, following the optimal
#   centralized controller (datasetTrajectories/)
# - Videos for some of the learned trajectories following the controles 
#   learned by each model (learnedTrajectories/)

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import alegnn.utils.dataTools as dataTools
import alegnn.utils.graphML as gml
import alegnn.modules.architecturesTime as architTime
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation

#\\\ Separate functions:
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed

# Start measuring time
startRunTime = datetime.datetime.now()

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

thisFilename = 'flockingGNN' # This is the general name of all related files

nAgents = 50 # Number of agents at training time

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-%03d-' % nAgents + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# Create the file where all the (hyper)parameters and results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########

useGPU = True # If true, and GPU is available, use it.

nAgentsMax = nAgents # Maximum number of agents to test the solution
nSimPoints = 1 # Number of simulations between nAgents and nAgentsMax
    # At test time, the architectures trained on nAgents will be tested on a
    # varying number of agents, starting at nAgents all the way to nAgentsMax;
    # the number of simulations for different number of agents is given by
    # nSimPoints, i.e. if nAgents = 50, nAgentsMax = 100 and nSimPoints = 3, 
    # then the architectures are trained on 50, 75 and 100 agents.
commRadius = 2. # Communication radius
repelDist = 1. # Minimum distance before activating repelling potential
nTrain = 400 # Number of training samples
nValid = 20 # Number of valid samples
nTest = 20 # Number of testing samples
duration = 2. # Duration of the trajectory
samplingTime = 0.01 # Sampling time
initGeometry = 'circular' # Geometry of initial positions
initVelValue = 3. # Initial velocities are samples from an interval
    # [-initVelValue, initVelValue]
initMinDist = 0.1 # No two agents are located at a distance less than this
accelMax = 10. # This is the maximum value of acceleration allowed

nRealizations = 10 # Number of data realizations
    # How many times we repeat the experiment

#\\\ Save values:
writeVarValues(varsFile,
               {'nAgents': nAgents,
                'nAgentsMax': nAgentsMax,
                'nSimPoints': nSimPoints,
                'commRadius': commRadius,
                'repelDist': repelDist,
                'nTrain': nTrain,
                'nValid': nValid,
                'nTest': nTest,
                'duration': duration,
                'samplingTime': samplingTime,
                'initGeometry': initGeometry,
                'initVelValue': initVelValue,
                'initMinDist': initMinDist,
                'accelMax': accelMax,
                'nRealizations': nRealizations,
                'useGPU': useGPU})

############
# TRAINING #
############

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.0005 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.MSELoss

#\\\ Training algorithm
trainer = training.TrainerFlocking

#\\\ Evaluation algorithm
evaluator = evaluation.evaluateFlocking

#\\\ Overall training options
probExpert = 0.993 # Probability of choosing the expert in DAGger
#DAGgerType = 'fixedBatch' # 'replaceTimeBatch', 'randomEpoch'
nEpochs = 30 # Number of epochs
batchSize = 20 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'optimizationAlgorithm': optimAlg,
                'learningRate': learningRate,
                'beta1': beta1,
                'beta2': beta2,
                'lossFunction': lossFunction,
                'trainer': trainer,
                'evaluator': evaluator,
                'probExpert': probExpert,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. Do not forget to add the name of the architecture
# to modelList.

# If the hyperparameter dictionary is called 'hParams' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N'
# variable after it has been coded).

# The name of the keys in the hyperparameter dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

#nFeatures = 32 # Number of features in all architectures
#nFilterTaps = 4 # Number of filter taps in all architectures
# [[The hyperparameters are for each architecture, and they were chosen 
#   following the results of the hyperparameter search]]
nonlinearityHidden = torch.tanh
nonlinearityOutput = torch.tanh
nonlinearity = nn.Tanh # Chosen nonlinearity for nonlinear architectures

# Select desired architectures
doLocalFlt = True # Local filter (no nonlinearity)
doLocalGNN = True # Local GNN (include nonlinearity)
doDlAggGNN = True
doGraphRNN = True

modelList = []

#\\\\\\\\\\\\\\\\\\
#\\\ FIR FILTER \\\
#\\\\\\\\\\\\\\\\\\

if doLocalFlt:

    #\\\ Basic parameters for the Local Filter architecture

    hParamsLocalFlt = {} # Hyperparameters (hParams) for the Local Filter

    hParamsLocalFlt['name'] = 'LocalFlt'
    # Chosen architecture
    hParamsLocalFlt['archit'] = architTime.LocalGNN_DB
    hParamsLocalFlt['device'] = 'cuda:0' \
                                    if (useGPU and torch.cuda.is_available()) \
                                    else 'cpu'

    # Graph convolutional parameters
    hParamsLocalFlt['dimNodeSignals'] = [6, 32] # Features per layer
    hParamsLocalFlt['nFilterTaps'] = [4] # Number of filter taps
    hParamsLocalFlt['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsLocalFlt['nonlinearity'] = gml.NoActivation # Selected nonlinearity
        # is affected by the summary
    # Readout layer: local linear combination of features
    hParamsLocalFlt['dimReadout'] = [2] # Dimension of the fully connected
        # layers after the FIR filter layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor 
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    hParamsLocalFlt['dimEdgeFeatures'] = 1 # Scalar edge weights

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsLocalFlt)
    modelList += [hParamsLocalFlt['name']]
    
#\\\\\\\\\\\\\\\\\
#\\\ LOCAL GNN \\\
#\\\\\\\\\\\\\\\\\

if doLocalGNN:

    #\\\ Basic parameters for the Local GNN architecture

    hParamsLocalGNN = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

    hParamsLocalGNN['name'] = 'LocalGNN'
    # Chosen architecture
    hParamsLocalGNN['archit'] = architTime.LocalGNN_DB
    hParamsLocalGNN['device'] = 'cuda:0' \
                                    if (useGPU and torch.cuda.is_available()) \
                                    else 'cpu'

    # Graph convolutional parameters
    hParamsLocalGNN['dimNodeSignals'] = [6, 64] # Features per layer
    hParamsLocalGNN['nFilterTaps'] = [3] # Number of filter taps
    hParamsLocalGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsLocalGNN['nonlinearity'] = nonlinearity # Selected nonlinearity
        # is affected by the summary
    # Readout layer: local linear combination of features
    hParamsLocalGNN['dimReadout'] = [2] # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor 
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    hParamsLocalGNN['dimEdgeFeatures'] = 1 # Scalar edge weights

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsLocalGNN)
    modelList += [hParamsLocalGNN['name']]
    
#\\\\\\\\\\\\\\\\\\\\\\\
#\\\ AGGREGATION GNN \\\
#\\\\\\\\\\\\\\\\\\\\\\\

if doDlAggGNN:

    #\\\ Basic parameters for the Aggregation GNN architecture

    hParamsDAGNN1Ly = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

    hParamsDAGNN1Ly['name'] = 'DAGNN1Ly'
    # Chosen architecture
    hParamsDAGNN1Ly['archit'] = architTime.AggregationGNN_DB
    hParamsDAGNN1Ly['device'] = 'cuda:0' \
                                    if (useGPU and torch.cuda.is_available()) \
                                    else 'cpu'

    # Graph convolutional parameters
    hParamsDAGNN1Ly['dimFeatures'] = [6] # Features per layer
    hParamsDAGNN1Ly['nFilterTaps'] = [] # Number of filter taps
    hParamsDAGNN1Ly['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsDAGNN1Ly['nonlinearity'] = nonlinearity # Selected nonlinearity
        # is affected by the summary
    hParamsDAGNN1Ly['poolingFunction'] = gml.NoPool
    hParamsDAGNN1Ly['poolingSize'] = []
    # Readout layer: local linear combination of features
    hParamsDAGNN1Ly['dimReadout'] = [64, 2] # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor 
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    hParamsDAGNN1Ly['dimEdgeFeatures'] = 1 # Scalar edge weights
    hParamsDAGNN1Ly['nExchanges'] = 2 - 1

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsDAGNN1Ly)
    modelList += [hParamsDAGNN1Ly['name']]
    
#\\\\\\\\\\\\\\\\\
#\\\ GRAPH RNN \\\
#\\\\\\\\\\\\\\\\\

if doGraphRNN:

    #\\\ Basic parameters for the Graph RNN architecture

    hParamsGraphRNN = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

    hParamsGraphRNN['name'] = 'GraphRNN'
    # Chosen architecture
    hParamsGraphRNN['archit'] = architTime.GraphRecurrentNN_DB
    hParamsGraphRNN['device'] = 'cuda:0' \
                                    if (useGPU and torch.cuda.is_available()) \
                                    else 'cpu'

    # Graph convolutional parameters
    hParamsGraphRNN['dimInputSignals'] = 6 # Features per layer
    hParamsGraphRNN['dimOutputSignals'] = 64
    hParamsGraphRNN['dimHiddenSignals'] = 64
    hParamsGraphRNN['nFilterTaps'] = [3] * 2 # Number of filter taps
    hParamsGraphRNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsGraphRNN['nonlinearityHidden'] = nonlinearityHidden
    hParamsGraphRNN['nonlinearityOutput'] = nonlinearityOutput
    hParamsGraphRNN['nonlinearityReadout'] = nonlinearity
    # Readout layer: local linear combination of features
    hParamsGraphRNN['dimReadout'] = [2] # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor 
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    hParamsGraphRNN['dimEdgeFeatures'] = 1 # Scalar edge weights

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGraphRNN)
    modelList += [hParamsGraphRNN['name']]

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 1 # After how many training steps, print the partial results
#   0 means to never print partial results while training
xAxisMultiplierTrain = 10 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 2 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers
videoSpeed = 0.5 # Slow down by half to show transitions
nVideos = 3 # Number of videos to save

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval,
                'figSize': figSize,
                'lineWidth': lineWidth,
                'markerShape': markerShape,
                'markerSize': markerSize,
                'videoSpeed': videoSpeed,
                'nVideos': nVideos})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ If CUDA is selected, empty cache:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

#\\\ Notify of processing units
if doPrint:
    print("Selected devices:")
    for thisModel in modelList:
        hParamsDict = eval('hParams' + thisModel)
        print("\t%s: %s" % (thisModel, hParamsDict['device']))

#\\\ Logging options
if doLogging:
    # If logging is on, load the tensorboard visualizer and initialize it
    from alegnn.utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')
    
#\\\ Number of agents at test time
nAgentsTest = np.linspace(nAgents, nAgentsMax, num = nSimPoints,dtype = np.int)
nAgentsTest = np.unique(nAgentsTest).tolist()
nSimPoints = len(nAgentsTest)
writeVarValues(varsFile, {'nAgentsTest': nAgentsTest}) # Save list

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# The first list is one for each value of nAgents that we want to simulate 
# (i.e. these are test results, so if we test for different number of agents,
# we need to save the results for each of them). Each element in the list will
# be a dictionary (i.e. for each testing case, we have a dictionary).
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
# We're saving the cost of the full trajectory, as well as the cost at the end
# instant.
costBestFull = [None] * nSimPoints
costBestEnd = [None] * nSimPoints
costLastFull = [None] * nSimPoints
costLastEnd = [None] * nSimPoints
costOptFull = [None] * nSimPoints
costOptEnd = [None] * nSimPoints
for n in range(nSimPoints):
    costBestFull[n] = {} # Accuracy for the best model (full trajectory)
    costBestEnd[n] = {} # Accuracy for the best model (end time)
    costLastFull[n] = {} # Accuracy for the last model
    costLastEnd[n] = {} # Accuracy for the last model
    for thisModel in modelList: # Create an element for each split realization,
        costBestFull[n][thisModel] = [None] * nRealizations
        costBestEnd[n][thisModel] = [None] * nRealizations
        costLastFull[n][thisModel] = [None] * nRealizations
        costLastEnd[n][thisModel] = [None] * nRealizations
    costOptFull[n] = [None] * nRealizations # Accuracy for optimal controller
    costOptEnd[n] = [None] * nRealizations # Accuracy for optimal controller

if doFigs:
    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    evalValid = {}
    # Initialize the splits dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nRealizations
        evalValid[thisModel] = [None] * nRealizations


####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of these options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doLogging:
    trainingOptions['logger'] = logger
if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

# And in case each model has specific training options (aka 'DAGger'), then
# we create a separate dictionary per model.

trainingOptsPerModel= {}

# Create relevant dirs: we need directories to save the videos of the dataset
# that involve the optimal centralized controllers, and we also need videos
# for the learned trajectory of each model. Note that all of these depend on
# each realization, so we will be saving videos for each realization.
# Here, we create all those directories.
datasetTrajectoryDir = os.path.join(saveDir,'datasetTrajectories')
if not os.path.exists(datasetTrajectoryDir):
    os.makedirs(datasetTrajectoryDir)
    
datasetTrainTrajectoryDir = os.path.join(datasetTrajectoryDir,'train')
if not os.path.exists(datasetTrainTrajectoryDir):
    os.makedirs(datasetTrainTrajectoryDir)
    
datasetTestTrajectoryDir = os.path.join(datasetTrajectoryDir,'test')
if not os.path.exists(datasetTestTrajectoryDir):
    os.makedirs(datasetTestTrajectoryDir)

datasetTestAgentTrajectoryDir = [None] * nSimPoints
for n in range(nSimPoints):    
    datasetTestAgentTrajectoryDir[n] = os.path.join(datasetTestTrajectoryDir,
                                                    '%03d' % nAgentsTest[n])
    
if nRealizations > 1:
    datasetTrainTrajectoryDirOrig = datasetTrainTrajectoryDir
    datasetTestAgentTrajectoryDirOrig = datasetTestAgentTrajectoryDir.copy()

#%%##################################################################
#                                                                   #
#                    DATA SPLIT REALIZATION                         #
#                                                                   #
#####################################################################

# Start generating a new data realization for each number of total realizations

for realization in range(nRealizations):

    # On top of the rest of the training options, we pass the identification
    # of this specific data split realization.

    if nRealizations > 1:
        trainingOptions['realizationNo'] = realization
        
        # Create new directories (specific for this realization)
        datasetTrainTrajectoryDir = os.path.join(datasetTrainTrajectoryDirOrig,
                                                 '%03d' % realization)
        if not os.path.exists(datasetTrainTrajectoryDir):
            os.makedirs(datasetTrainTrajectoryDir)
            
        for n in range(nSimPoints):
            datasetTestAgentTrajectoryDir[n] = os.path.join(
                                          datasetTestAgentTrajectoryDirOrig[n],
                                          '%03d' % realization)
            if not os.path.exists(datasetTestAgentTrajectoryDir[n]):
                os.makedirs(datasetTestAgentTrajectoryDir[n])

    if doPrint:
        print("", flush = True)

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    ############
    # DATASETS #
    ############

    if doPrint:
        print("Generating data", end = '')
        if nRealizations > 1:
            print(" for realization %d" % realization, end = '')
        print("...", flush = True)

    #   Generate the dataset
    data = dataTools.Flocking(
                # Structure
                nAgents,
                commRadius,
                repelDist,
                # Samples
                nTrain,
                nValid,
                1, # We do not care about testing, we will re-generate the
                   # dataset for testing
                # Time
                duration,
                samplingTime,
                # Initial conditions
                initGeometry = initGeometry,
                initVelValue = initVelValue,
                initMinDist = initMinDist,
                accelMax = accelMax)

    ###########
    # PREVIEW #
    ###########

    if doPrint:
        print("Preview data", end = '')
        if nRealizations > 1:
            print(" for realization %d" % realization, end = '')
        print("...", flush = True)

    # Generate the videos
    data.saveVideo(datasetTrainTrajectoryDir, # Where to save them
                    data.pos['train'], # Which positions to plot
                    nVideos, # Number of videos to create
                    commGraph = data.commGraph['train'], # Graph to plot
                    vel = data.vel['train'], # Velocity arrows to plot
                    videoSpeed = videoSpeed) # Change speed of animation

    #%%##################################################################
    #                                                                   #
    #                    MODELS INITIALIZATION                          #
    #                                                                   #
    #####################################################################

    # This is the dictionary where we store the models (in a model.Model
    # class).
    modelsGNN = {}

    # If a new model is to be created, it should be called for here.

    if doPrint:
        print("Model initialization...", flush = True)

    for thisModel in modelList:

        # Get the corresponding parameter dictionary
        hParamsDict = deepcopy(eval('hParams' + thisModel))
        # and training options
        trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)

        # Now, this dictionary has all the hyperparameters that we need to pass
        # to the architecture, but it also has the 'name' and 'archit' that
        # we do not need to pass them. So we are going to get them out of
        # the dictionary
        thisName = hParamsDict.pop('name')
        callArchit = hParamsDict.pop('archit')
        thisDevice = hParamsDict.pop('device')
        # If there's a specific DAGger type, pop it out now
        if 'DAGgerType' in hParamsDict.keys() \
                                        and 'probExpert' in hParamsDict.keys():
            trainingOptsPerModel[thisModel]['probExpert'] = \
                                                  hParamsDict.pop('probExpert')
            trainingOptsPerModel[thisModel]['DAGgerType'] = \
                                                  hParamsDict.pop('DAGgerType')

        # If more than one graph or data realization is going to be carried out,
        # we are going to store all of thos models separately, so that any of
        # them can be brought back and studied in detail.
        if nRealizations > 1:
            thisName += 'G%02d' % realization

        if doPrint:
            print("\tInitializing %s..." % thisName,
                  end = ' ',flush = True)

        ##############
        # PARAMETERS #
        ##############

        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisOptimAlg = optimAlg
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2

        ################
        # ARCHITECTURE #
        ################

        thisArchit = callArchit(**hParamsDict)
        thisArchit.to(thisDevice)

        #############
        # OPTIMIZER #
        #############

        if thisOptimAlg == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate,
                                   betas = (beta1, beta2))
        elif thisOptimAlg == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(),
                                  lr = learningRate)
        elif thisOptimAlg == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)

        ########
        # LOSS #
        ########

        thisLossFunction = lossFunction()
        
        ###########
        # TRAINER #
        ###########

        thisTrainer = trainer
        
        #############
        # EVALUATOR #
        #############

        thisEvaluator = evaluator

        #########
        # MODEL #
        #########

        modelCreated = model.Model(thisArchit,
                                   thisLossFunction,
                                   thisOptim,
                                   thisTrainer,
                                   thisEvaluator,
                                   thisDevice,
                                   thisName,
                                   saveDir)

        modelsGNN[thisName] = modelCreated

        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisOptimizationAlgorithm': thisOptimAlg,
                        'thisTrainer': thisTrainer,
                        'thisEvaluator': thisEvaluator,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2})

        if doPrint:
            print("OK")

    #%%##################################################################
    #                                                                   #
    #                    TRAINING                                       #
    #                                                                   #
    #####################################################################


    ############
    # TRAINING #
    ############

    print("")

    for thisModel in modelsGNN.keys():

        if doPrint:
            print("Training model %s..." % thisModel)
            
        for m in modelList:
            if m in thisModel:
                modelName = m

        thisTrainVars = modelsGNN[thisModel].train(data,
                                                   nEpochs,
                                                   batchSize,
                                                   **trainingOptsPerModel[m])

        if doFigs:
        # Find which model to save the results (when having multiple
        # realizations)
            for m in modelList:
                if m in thisModel:
                    lossTrain[m][realization] = thisTrainVars['lossTrain']
                    evalValid[m][realization] = thisTrainVars['evalValid']
    # And we also need to save 'nBatch' but is the same for all models, so
    if doFigs:
        nBatches = thisTrainVars['nBatches']

    #%%##################################################################
    #                                                                   #
    #                    EVALUATION                                     #
    #                                                                   #
    #####################################################################

    # Now that the model has been trained, we evaluate them on the test
    # samples.

    # We have two versions of each model to evaluate: the one obtained
    # at the best result of the validation step, and the last trained model.
        
    for n in range(nSimPoints):
        
        if doPrint:
            print("")
            print("[%3d Agents] Generating test set" % nAgentsTest[n],
                  end = '')
            if nRealizations > 1:
                print(" for realization %d" % realization, end = '')
            print("...", flush = True)

        #   Load the data, which will give a specific split
        dataTest = dataTools.Flocking(
                        # Structure
                        nAgentsTest[n],
                        commRadius,
                        repelDist,
                        # Samples
                        1, # We don't care about training
                        1, # nor validation
                        nTest,
                        # Time
                        duration,
                        samplingTime,
                        # Initial conditions
                        initGeometry = initGeometry,
                        initVelValue = initVelValue,
                        initMinDist = initMinDist,
                        accelMax = accelMax)
    
        ###########
        # OPTIMAL #
        ###########
        
        #\\\ PREVIEW
        #\\\\\\\\\\\
        
        # Save videos for the optimal trajectories of the test set (before it
        # was for the otpimal trajectories of the training set)
        
        posTest = dataTest.getData('pos', 'test')
        velTest = dataTest.getData('vel', 'test')
        commGraphTest = dataTest.getData('commGraph', 'test')
    
        if doPrint:
            print("[%3d Agents] Preview data"  % nAgentsTest[n], end = '')
            if nRealizations > 1:
                print(" for realization %d" % realization, end = '')
            print("...", flush = True)
    
        dataTest.saveVideo(datasetTestAgentTrajectoryDir[n],
                           posTest,
                           nVideos,
                           commGraph = commGraphTest,
                           vel = velTest,
                           videoSpeed = videoSpeed)
        
        #\\\ EVAL
        #\\\\\\\\
        
        # Get the cost for the optimal trajectories
        
        # Full trajectory
        costOptFull[n][realization] = dataTest.evaluate(vel = velTest)
        
        # Last time instant
        costOptEnd[n][realization] = dataTest.evaluate(vel = velTest[:,-1:,:,:])
        
        writeVarValues(varsFile,
                   {'costOptFull%03dR%02d' % (nAgentsTest[n],realization):
                                                     costOptFull[n][realization],
                    'costOptEnd%04dR%02d' % (nAgentsTest[n],realization):
                                                     costOptEnd[n][realization]})
        
        del posTest, velTest, commGraphTest
        
        ##########
        # MODELS #
        ##########
    
        for thisModel in modelsGNN.keys():
    
            if doPrint:
                print("[%3d Agents] Evaluating model %s" % \
                                         (nAgentsTest[n], thisModel), end = '')
                if nRealizations > 1:
                    print(" for realization %d" % realization, end = '')
                print("...", flush = True)
                
            addKW = {}
            addKW['nVideos'] = nVideos
            addKW['graphNo'] = nAgentsTest[n]
            if nRealizations > 1:
                addKW['realizationNo'] = realization
                
            thisEvalVars = modelsGNN[thisModel].evaluate(dataTest, **addKW)
    
            thisCostBestFull = thisEvalVars['costBestFull']
            thisCostBestEnd = thisEvalVars['costBestEnd']
            thisCostLastFull = thisEvalVars['costLastFull']
            thisCostLastEnd = thisEvalVars['costLastEnd']
            
            # Save values
            writeVarValues(varsFile,
                   {'costBestFull%s%03dR%02d' % \
                                       (thisModel, nAgentsTest[n], realization):
                                                                thisCostBestFull,
                    'costBestEnd%s%04dR%02d' % \
                                       (thisModel, nAgentsTest[n], realization):
                                                                 thisCostBestEnd,
                    'costLastFull%s%03dR%02d' % \
                                       (thisModel, nAgentsTest[n], realization):
                                                                thisCostLastFull,
                    'costLastEnd%s%04dR%02d' % \
                                       (thisModel, nAgentsTest[n], realization):
                                                                thisCostLastEnd})
    
            # Find which model to save the results (when having multiple
            # realizations)
            for m in modelList:
                if m in thisModel:
                    costBestFull[n][m][realization] = thisCostBestFull
                    costBestEnd[n][m][realization] = thisCostBestEnd
                    costLastFull[n][m][realization] = thisCostLastFull
                    costLastEnd[n][m][realization] = thisCostLastEnd


############################
# FINAL EVALUATION RESULTS #
############################
                    
meanCostBestFull = [None] * nSimPoints # Mean across data splits
meanCostBestEnd = [None] * nSimPoints # Mean across data splits
meanCostLastFull = [None] * nSimPoints # Mean across data splits
meanCostLastEnd = [None] * nSimPoints # Mean across data splits
stdDevCostBestFull = [None] * nSimPoints # Standard deviation across data splits
stdDevCostBestEnd = [None] * nSimPoints # Standard deviation across data splits
stdDevCostLastFull = [None] * nSimPoints # Standard deviation across data splits
stdDevCostLastEnd = [None] * nSimPoints # Standard deviation across data splits
meanCostOptFull = [None] * nSimPoints
stdDevCostOptFull = [None] * nSimPoints
meanCostOptEnd = [None] * nSimPoints
stdDevCostOptEnd = [None] * nSimPoints
                    
for n in range(nSimPoints):

    # Now that we have computed the accuracy of all runs, we can obtain a final
    # result (mean and standard deviation)
    
    meanCostBestFull[n] = {} # Mean across data splits
    meanCostBestEnd[n] = {} # Mean across data splits
    meanCostLastFull[n] = {} # Mean across data splits
    meanCostLastEnd[n] = {} # Mean across data splits
    stdDevCostBestFull[n] = {} # Standard deviation across data splits
    stdDevCostBestEnd[n] = {} # Standard deviation across data splits
    stdDevCostLastFull[n] = {} # Standard deviation across data splits
    stdDevCostLastEnd[n] = {} # Standard deviation across data splits
    
    if doPrint:
        print("\n[%3d Agents] Final evaluations (%02d data splits)" % \
                                               (nAgentsTest[n], nRealizations))
        
    costOptFull[n] = np.array(costOptFull[n])
    meanCostOptFull[n] = np.mean(costOptFull[n])
    stdDevCostOptFull[n] = np.std(costOptFull[n])
    costOptEnd[n] = np.array(costOptEnd[n])
    meanCostOptEnd[n] = np.mean(costOptEnd[n])
    stdDevCostOptEnd[n] = np.std(costOptEnd[n])
    
    if doPrint:
        print("\t%8s: %8.4f (+-%6.4f) [Optm/Full]" % (
                'Optimal',
                meanCostOptFull[n],
                stdDevCostOptFull[n]))
        print("\t%9s %8.4f (+-%6.4f) [Optm/End ]" % (
                '',
                meanCostOptEnd[n],
                stdDevCostOptEnd[n]))
        
    # Save values
    writeVarValues(varsFile,
               {'meanCostOptFull%03d' % nAgentsTest[n]: meanCostOptFull[n],
                'stdDevCostOptFull%03d' % nAgentsTest[n]: stdDevCostOptFull[n],
                'meanCostOptEnd%04d' % nAgentsTest[n]: meanCostOptEnd[n],
                'stdDevCostOptEnd%04d' % nAgentsTest[n]: stdDevCostOptEnd[n]})
    
    for thisModel in modelList:
        # Convert the lists into a nDataSplits vector
        costBestFull[n][thisModel] = np.array(costBestFull[n][thisModel])
        costBestEnd[n][thisModel] = np.array(costBestEnd[n][thisModel])
        costLastFull[n][thisModel] = np.array(costLastFull[n][thisModel])
        costLastEnd[n][thisModel] = np.array(costLastEnd[n][thisModel])
    
        # And now compute the statistics (across graphs)
        meanCostBestFull[n][thisModel] = np.mean(costBestFull[n][thisModel])
        meanCostBestEnd[n][thisModel] = np.mean(costBestEnd[n][thisModel])
        meanCostLastFull[n][thisModel] = np.mean(costLastFull[n][thisModel])
        meanCostLastEnd[n][thisModel] = np.mean(costLastEnd[n][thisModel])
        stdDevCostBestFull[n][thisModel] = np.std(costBestFull[n][thisModel])
        stdDevCostBestEnd[n][thisModel] = np.std(costBestEnd[n][thisModel])
        stdDevCostLastFull[n][thisModel] = np.std(costLastFull[n][thisModel])
        stdDevCostLastEnd[n][thisModel] = np.std(costLastEnd[n][thisModel])
    
        # And print it:
        if doPrint:
            print(
              "\t%s: %8.4f (+-%6.4f) [Best/Full] %8.4f (+-%6.4f) [Last/Full]"%(
                    thisModel,
                    meanCostBestFull[n][thisModel],
                    stdDevCostBestFull[n][thisModel],
                    meanCostLastFull[n][thisModel],
                    stdDevCostLastFull[n][thisModel]))
            print(
              "\t%9s %8.4f (+-%6.4f) [Best/End ] %8.4f (+-%6.4f) [Last/End ]"%(
                    '',
                    meanCostBestEnd[n][thisModel],
                    stdDevCostBestEnd[n][thisModel],
                    meanCostLastEnd[n][thisModel],
                    stdDevCostLastEnd[n][thisModel]))
    
        # Save values
        writeVarValues(varsFile,
                   {'meanAccBestFull%s%03d' % (thisModel, nAgentsTest[n]): 
                                               meanCostBestFull[n][thisModel],
                    'stdDevAccBestFull%s%03d' % (thisModel, nAgentsTest[n]): 
                                               stdDevCostBestFull[n][thisModel],
                    'meanAccBestEnd%s%04d' % (thisModel, nAgentsTest[n]): 
                                               meanCostBestEnd[n][thisModel],
                    'stdDevAccBestEnd%s%04d' % (thisModel, nAgentsTest[n]): 
                                               stdDevCostBestEnd[n][thisModel],
                    'meanAccLastFull%s%03d' % (thisModel, nAgentsTest[n]): 
                                               meanCostLastFull[n][thisModel],
                    'stdDevAccLastFull%s%03d' % (thisModel, nAgentsTest[n]): 
                                               stdDevCostLastFull[n][thisModel],
                    'meanAccLastEnd%s%04d' % (thisModel, nAgentsTest[n]): 
                                               meanCostLastEnd[n][thisModel],
                    'stdDevAccLastEnd%s%04d' % (thisModel, nAgentsTest[n]): 
                                               stdDevCostLastEnd[n][thisModel]})
            
    # Save the printed info into the .txt file as well
    with open(varsFile, 'a+') as file:
        file.write("\n[%3d Agents] Final evaluations (%02d data splits)" % \
                                               (nAgentsTest[n], nRealizations))
        file.write("\t%8s: %8.4f (+-%6.4f) [Optm/Full]" % (
                   'Optimal',
                   meanCostOptFull[n],
                   stdDevCostOptFull[n]))
        file.write("\t%9s %8.4f (+-%6.4f) [Optm/End ]" % (
                   '',
                   meanCostOptEnd[n],
                   stdDevCostOptEnd[n]))
        for thisModel in modelList:
            file.write(
              "\t%s: %8.4f (+-%6.4f) [Best/Full] %8.4f (+-%6.4f) [Last/Full]"%(
                    thisModel,
                    meanCostBestFull[n][thisModel],
                    stdDevCostBestFull[n][thisModel],
                    meanCostLastFull[n][thisModel],
                    stdDevCostLastFull[n][thisModel]))
            file.write(
              "\t%9s %8.4f (+-%6.4f) [Best/End ] %8.4f (+-%6.4f) [Last/End ]"%(
                    '',
                    meanCostBestEnd[n][thisModel],
                    stdDevCostBestEnd[n][thisModel],
                    meanCostLastEnd[n][thisModel],
                    stdDevCostLastEnd[n][thisModel]))
        file.write('\n')

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################

# Finally, we might want to plot several quantities of interest

if doFigs:

    ###################
    # DATA PROCESSING #
    ###################

    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    meanLossTrain = {}
    meanEvalValid = {}
    stdDevLossTrain = {}
    stdDevEvalValid = {}
    # Initialize the variables
    for thisModel in modelList:
        # Transform into np.array
        lossTrain[thisModel] = np.array(lossTrain[thisModel])
        evalValid[thisModel] = np.array(evalValid[thisModel])
        # Each of one of these variables should be of shape
        # nDataSplits x numberOfTrainingSteps
        # And compute the statistics
        meanLossTrain[thisModel] = np.mean(lossTrain[thisModel], axis = 0)
        meanEvalValid[thisModel] = np.mean(evalValid[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = np.std(lossTrain[thisModel], axis = 0)
        stdDevEvalValid[thisModel] = np.std(evalValid[thisModel], axis = 0)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    #   Pickle, first:
    varsPickle = {}
    varsPickle['nEpochs'] = nEpochs
    varsPickle['nBatches'] = nBatches
    varsPickle['meanLossTrain'] = meanLossTrain
    varsPickle['stdDevLossTrain'] = stdDevLossTrain
    varsPickle['meanEvalValid'] = meanEvalValid
    varsPickle['stdDevEvalValid'] = stdDevEvalValid
    with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(varsPickle, figVarsFile)

    ########
    # PLOT #
    ########

    # Compute the x-axis
    xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
    xValid = np.arange(0, nEpochs * nBatches, \
                          validationInterval*xAxisMultiplierValid)

    # If we do not want to plot all the elements (to avoid overcrowded plots)
    # we need to recompute the x axis and take those elements corresponding
    # to the training steps we want to plot
    if xAxisMultiplierTrain > 1:
        # Actual selected samples
        selectSamplesTrain = xTrain
        # Go and fetch tem
        for thisModel in modelList:
            meanLossTrain[thisModel] = meanLossTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel]\
                                                        [selectSamplesTrain]
    # And same for the validation, if necessary.
    if xAxisMultiplierValid > 1:
        selectSamplesValid = np.arange(0, len(meanEvalValid[thisModel]), \
                                       xAxisMultiplierValid)
        for thisModel in modelList:
            meanEvalValid[thisModel] = meanEvalValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevEvalValid[thisModel] = stdDevEvalValid[thisModel]\
                                                        [selectSamplesValid]

    #\\\ LOSS (Training and validation) for EACH MODEL
    for key in meanLossTrain.keys():
        lossFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training'])
        plt.title(r'%s' % key)
        lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
                        bbox_inches = 'tight')
        plt.close(fig = lossFig)

    #\\\ Cost (Training and validation) for EACH MODEL
    for key in meanEvalValid.keys():
        accFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Cost')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        accFig.savefig(os.path.join(saveDirFigs,'eval%s.pdf' % key),
                        bbox_inches = 'tight')
        plt.close(fig = accFig)

    # LOSS (training) for ALL MODELS
    allLossTrain = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanLossTrain.keys():
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Loss')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanLossTrain.keys()))
    allLossTrain.savefig(os.path.join(saveDirFigs,'allLossTrain.pdf'),
                    bbox_inches = 'tight')
    plt.close(fig = allLossTrain)

    # Cost (validation) for ALL MODELS
    allEvalValid = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanEvalValid.keys():
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Cost')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanEvalValid.keys()))
    allEvalValid.savefig(os.path.join(saveDirFigs,'allEvalValid.pdf'),
                    bbox_inches = 'tight')
    plt.close(fig = allEvalValid)

# Finish measuring time
endRunTime = datetime.datetime.now()

totalRunTime = abs(endRunTime - startRunTime)
totalRunTimeH = int(divmod(totalRunTime.total_seconds(), 3600)[0])
totalRunTimeM, totalRunTimeS = \
               divmod(totalRunTime.total_seconds() - totalRunTimeH * 3600., 60)
totalRunTimeM = int(totalRunTimeM)

if doPrint:
    print(" ")
    print("Simulation started: %s" %startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Simulation ended:   %s" % endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                         totalRunTimeM,
                                         totalRunTimeS))
    
# And save this info into the .txt file as well
with open(varsFile, 'a+') as file:
    file.write("Simulation started: %s\n" % 
                                     startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Simulation ended:   %s\n" % 
                                       endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                              totalRunTimeM,
                                              totalRunTimeS))
