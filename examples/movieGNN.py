# 2019/04/10~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu

# Test a movie recommendation problem. The nodes are either items or users
# and the edges are rating similarities estimated by a Pearson correlation
# coefficient (either rating similarities between items or rating similarities
# between users). The graph signal defined on top of this graph are the
# ratings given to items by a specific user (if the nodes are items) or the
# ratings given by the users to a specific item (if the nodes are users).
# The objective is to estimate the rating on a target node(s), that is, an
# interpolation problem of the graph signal at a target node(s).

# Outputs:
# - Text file with all the hyperparameters selected for the run and the 
#   corresponding results (hyperparameters.txt)
# - Pickle file with the random seeds of both torch and numpy for accurate
#   reproduction of results (randomSeedUsed.pkl)
# - The parameters of the trained models, for both the Best and the Last
#   instance of each model (savedModels/)
# - The figures of loss and evaluation through the training iterations for
#   each model (figs/ and trainVars/)
# - If selected, logs in tensorboardX certain useful training variables

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
import alegnn.utils.graphTools as graphTools
import alegnn.utils.dataTools
import alegnn.utils.graphML as gml
import alegnn.modules.architectures as archit
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation
import alegnn.modules.loss as loss

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

graphType = 'movie' # Graph type: 'user'-based or 'movie'-based
labelID = [50] # Which node to focus on (either a list or the str 'all')
# When 'movie': [1]: Toy Story, [50]: Star Wars, [258]: Contact,
# [100]: Fargo, [181]: Return of the Jedi, [294]: Liar, liar
if labelID == 'all':
    labelIDstr = 'all'
elif len(labelID) == 1:
    labelIDstr = '%03d' % labelID[0]
else:
    labelIDstr = ['%03d_' % i for i in labelID]
    labelIDstr = "".join(labelIDstr)
    labelIDstr = labelIDstr[0:-1]
thisFilename = 'movieGNN' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run
dataDir = os.path.join('datasets','movielens')

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + graphType + '-' + labelIDstr + '-' + today
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

ratioTrain = 0.9 # Ratio of training samples
ratioValid = 0.1 # Ratio of validation samples (out of the total training
# samples)
# Final split is:
#   nValidation = round(ratioValid * ratioTrain * nTotal)
#   nTrain = round((1 - ratioValid) * ratioTrain * nTotal)
#   nTest = nTotal - nTrain - nValidation
maxNodes = None # Maximum number of nodes (select the ones with the largest
#   number of ratings)
minRatings = 0 # Discard samples (rows and columns) with less than minRatings 
    # ratings
interpolateRatings = False # Interpolate ratings with nearest-neighbors rule
    # before feeding them into the GNN

nDataSplits = 10 # Number of data realizations
# Obs.: The built graph depends on the split between training, validation and
# testing. Therefore, we will run several of these splits and average across
# them, to obtain some result that is more robust to this split.
    
# Given that we build the graph from a training split selected at random, it
# could happen that it is disconnected, or directed, or what not. In other 
# words, we might want to force (by removing nodes) some useful characteristics
# on the graph
keepIsolatedNodes = True # If True keeps isolated nodes ---> FALSE
forceUndirected = True # If True forces the graph to be undirected
forceConnected = False # If True returns the largest connected component of the
    # graph as the main graph ---> TRUE
kNN = 10 # Number of nearest neighbors

maxDataPoints = None # None to consider all data points

#\\\ Save values:
writeVarValues(varsFile,
               {'labelID': labelID,
                'graphType': graphType,
                'ratioTrain': ratioTrain,
                'ratioValid': ratioValid,
                'maxNodes': maxNodes,
                'minRatings': minRatings,
                'interpolateRatings': interpolateRatings,
                'nDataSplits': nDataSplits,
                'keepIsolatedNodes': keepIsolatedNodes,
                'forceUndirected': forceUndirected,
                'forceConnected': forceConnected,
                'kNN': kNN,
                'maxDataPoints': maxDataPoints,
                'useGPU': useGPU})

############
# TRAINING #
############

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.005 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.SmoothL1Loss

#\\\ Overall training options
nEpochs = 40 # Number of epochs
batchSize = 5 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'optimAlg': optimAlg,
                'learningRate': learningRate,
                'beta1': beta1,
                'beta2': beta2,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# Just four architecture one- and two-layered Selection and Local GNN. The main
# difference is that the Local GNN is entirely local (i.e. the output is given
# by a linear combination of the features at a single node, instead of a final
# MLP layer combining the features at all nodes).
    
# Select desired architectures
doSelectionGNN = True
doLocalGNN = True

do1Layer = True
do2Layers = True

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. Do not forget to add the name of the architecture
# to modelList.

# If the model dictionary is called 'model' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N' 
# variable after it has been coded).

# The name of the keys in the model dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

modelList = []

#\\\\\\\\\\\\\\\\\\\\\
#\\\ SELECTION GNN \\\
#\\\\\\\\\\\\\\\\\\\\\

if doSelectionGNN:

    #\\\ Basic parameters for all the Selection GNN architectures
    
    modelSelGNN = {} # Model parameters for the Selection GNN (SelGNN)
    modelSelGNN['name'] = 'SelGNN'
    modelSelGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Chosen architecture
    modelSelGNN['archit'] = archit.SelectionGNN
    # Graph convolutional parameters
    modelSelGNN['dimNodeSignals'] = [1, 64, 32] # Features per layer
    modelSelGNN['nFilterTaps'] = [5, 5] # Number of filter taps per layer
    modelSelGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    modelSelGNN['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    modelSelGNN['poolingFunction'] = gml.NoPool # Summarizing function
    modelSelGNN['nSelectedNodes'] = None # To be determined later on
    modelSelGNN['poolingSize'] = [1, 1] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Full MLP readout layer (this layer breaks the locality of the solution)
    modelSelGNN['dimLayersMLP'] = [1] # Dimension of the fully connected
        # layers after the GCN layers, we just need to output a single scalar
    # Graph structure
    modelSelGNN['GSO'] = None # To be determined later on, based on data
    modelSelGNN['order'] = None # Not used because there is no pooling
    
    #\\\ TRAINER

    modelSelGNN['trainer'] = training.Trainer
    
    #\\\ EVALUATOR
    
    modelSelGNN['evaluator'] = evaluation.evaluate
    

#\\\\\\\\\\\\
#\\\ MODEL 1: Selection GNN with 1 less layer
#\\\\\\\\\\\\

    modelSelGNN1Ly = deepcopy(modelSelGNN)

    modelSelGNN1Ly['name'] += '1Ly' # Name of the architecture
    
    modelSelGNN1Ly['dimNodeSignals'] = modelSelGNN['dimNodeSignals'][0:-1]
    modelSelGNN1Ly['nFilterTaps'] = modelSelGNN['nFilterTaps'][0:-1]
    modelSelGNN1Ly['poolingSize'] = modelSelGNN['poolingSize'][0:-1]

    #\\\ Save Values:
    writeVarValues(varsFile, modelSelGNN1Ly)
    modelList += [modelSelGNN1Ly['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 2: Selection GNN with all Layers
#\\\\\\\\\\\\

    modelSelGNN2Ly = deepcopy(modelSelGNN)

    modelSelGNN2Ly['name'] += '2Ly' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, modelSelGNN2Ly)
    modelList += [modelSelGNN2Ly['name']]

    
#\\\\\\\\\\\\\\\\\
#\\\ LOCAL GNN \\\
#\\\\\\\\\\\\\\\\\

if doLocalGNN:

    #\\\ Basic parameters for all the Local GNN architectures
    
    modelLclGNN = {} # Model parameters for the Local GNN (LclGNN)
    modelLclGNN['name'] = 'LclGNN'
    modelLclGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Chosen architecture
    modelLclGNN['archit'] = archit.LocalGNN
    # Graph convolutional parameters
    modelLclGNN['dimNodeSignals'] = [1, 64, 32] # Features per layer
    modelLclGNN['nFilterTaps'] = [5, 5] # Number of filter taps per layer
    modelLclGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    modelLclGNN['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    modelLclGNN['poolingFunction'] = gml.NoPool # Summarizing function
    modelLclGNN['nSelectedNodes'] = None # To be determined later on
    modelLclGNN['poolingSize'] = [1, 1] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Readout layer: local linear combination of features
    modelLclGNN['dimReadout'] = [1] # Dimension of the fully connected layers
        # after the GCN layers (map); this fully connected layer is applied only
        # at each node, without any further exchanges nor considering all nodes
        # at once, making the architecture entirely local.
    # Graph structure
    modelLclGNN['GSO'] = None # To be determined later on, based on data
    modelLclGNN['order'] = None # Not used because there is no pooling
    
    #\\\ TRAINER

    modelLclGNN['trainer'] = training.TrainerSingleNode
    
    #\\\ EVALUATOR
    
    modelLclGNN['evaluator'] = evaluation.evaluateSingleNode

#\\\\\\\\\\\\
#\\\ MODEL 3: Local GNN with 1 less layer
#\\\\\\\\\\\\

    modelLclGNN1Ly = deepcopy(modelLclGNN)

    modelLclGNN1Ly['name'] += '1Ly' # Name of the architecture
    
    modelLclGNN1Ly['dimNodeSignals'] = modelLclGNN['dimNodeSignals'][0:-1]
    modelLclGNN1Ly['nFilterTaps'] = modelLclGNN['nFilterTaps'][0:-1]
    modelLclGNN1Ly['poolingSize'] = modelLclGNN['poolingSize'][0:-1]

    #\\\ Save Values:
    writeVarValues(varsFile, modelLclGNN1Ly)
    modelList += [modelLclGNN1Ly['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 4: Local GNN with all Layers
#\\\\\\\\\\\\

    modelLclGNN2Ly = deepcopy(modelLclGNN)

    modelLclGNN2Ly['name'] += '2Ly' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, modelLclGNN2Ly)
    modelList += [modelLclGNN2Ly['name']]

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 5 # After how many training steps, print the partial results
#   0 means to never print partial results while training
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 20 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers

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
                'markerSize': markerSize})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Determine processing unit:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

#\\\ Notify of processing units
if doPrint:
    print("Selected devices:")
    for thisModel in modelList:
        modelDict = eval('model' + thisModel)
        print("\t%s: %s" % (thisModel, modelDict['device']))

#\\\ Logging options
if doLogging:
    # If logging is on, load the tensorboard visualizer and initialize it
    from alegnn.utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
costBest = {} # Cost for the best model (Evaluation cost: RMSE)
costLast = {} # Cost for the last model
for thisModel in modelList: # Create an element for each split realization,
    costBest[thisModel] = [None] * nDataSplits
    costLast[thisModel] = [None] * nDataSplits

if doFigs:
    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    costTrain = {}
    lossValid = {}
    costValid = {}
    # Initialize the splits dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nDataSplits
        costTrain[thisModel] = [None] * nDataSplits
        lossValid[thisModel] = [None] * nDataSplits
        costValid[thisModel] = [None] * nDataSplits

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

# And in case each model has specific training options, then we create a 
# separate dictionary per model.

trainingOptsPerModel= {}

#%%##################################################################
#                                                                   #
#                    DATA SPLIT REALIZATION                         #
#                                                                   #
#####################################################################

# Start generating a new data split for each of the number of data splits that
# we previously specified

for split in range(nDataSplits):

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    ############
    # DATASETS #
    ############
    
    if doPrint:
        print("Loading data", end = '')
        if nDataSplits > 1:
            print(" for split %d" % (split+1), end = '')
        print("...", end = ' ', flush = True)

    #   Load the data, which will give a specific split
    data = alegnn.utils.dataTools.MovieLens(graphType,  # 'user' or 'movies'
                                            labelID,  # ID of node to interpolate
                                            ratioTrain,  # ratio of training samples
                                            ratioValid,  # ratio of validation samples
                                            dataDir,  # directory where dataset is
                                            # Extra options
                                            keepIsolatedNodes,
                                            forceUndirected,
                                            forceConnected,
                                            kNN,  # Number of nearest neighbors
                                            maxNodes = maxNodes,
                                            maxDataPoints = maxDataPoints,
                                            minRatings = minRatings,
                                            interpolate = interpolateRatings)
    
    if doPrint:
        print("OK")

    #########
    # GRAPH #
    #########
    
    if doPrint:
        print("Setting up the graph...", end = ' ', flush = True)

    # Create graph
    adjacencyMatrix = data.getGraph()
    G = graphTools.Graph('adjacency', adjacencyMatrix.shape[0],
                         {'adjacencyMatrix': adjacencyMatrix})
    G.computeGFT() # Compute the GFT of the stored GSO

    # And re-update the number of nodes for changes in the graph (due to
    # enforced connectedness, for instance)
    nNodes = G.N

    # Once data is completely formatted and in appropriate fashion, change its
    # type to torch
    data.astype(torch.float64)
    # And the corresponding feature dimension that we will need to use
    data.expandDims() # Data are just graph signals, but the architectures 
        # require that the input signals are of the form B x F x N, so we need
        # to expand the middle dimensions to convert them from B x N to
        # B x 1 x N
    
    if doPrint:
        print("OK")

    #%%##################################################################
    #                                                                   #
    #                    MODELS INITIALIZATION                          #
    #                                                                   #
    #####################################################################

    # This is the dictionary where we store the models (in a model.Model
    # class, that is then passed to training).
    modelsGNN = {}

    # If a new model is to be created, it should be called for here.
    
    if doPrint:
        print("Model initialization...", flush = True)
        
    for thisModel in modelList:
        
        # Get the corresponding parameter dictionary
        modelDict = deepcopy(eval('model' + thisModel))
        # and training options
        trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)
        
        # Now, this dictionary has all the hyperparameters that we need to pass
        # to the architecture function, but it also has other keys that belong
        # to the more general model (like 'name' or 'device'), so we need to
        # extract them and save them in seperate variables for future use.
        thisName = modelDict.pop('name')
        callArchit = modelDict.pop('archit')
        thisDevice = modelDict.pop('device')
        thisTrainer = modelDict.pop('trainer')
        thisEvaluator = modelDict.pop('evaluator')
        
        # If more than one graph or data realization is going to be carried out,
        # we are going to store all of thos models separately, so that any of
        # them can be brought back and studied in detail.
        if nDataSplits > 1:
            thisName += 'G%02d' % split
            
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

        #\\\ Ordering
        S = G.S.copy()/np.max(np.real(G.E))
        # Do not forget to add the GSO to the input parameters of the archit
        modelDict['GSO'] = S
        # Add the number of nodes for the no-pooling part
        if '1Ly' in thisName:
            modelDict['nSelectedNodes'] = [nNodes]
        elif '2Ly' in thisName:
            modelDict['nSelectedNodes'] = [nNodes, nNodes]
        
        ################
        # ARCHITECTURE #
        ################

        thisArchit = callArchit(**modelDict)
        
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

        # Initialize the loss function
        thisLossFunction = loss.adaptExtraDimensionLoss(lossFunction)

        #########
        # MODEL #
        #########

        # Create the model
        modelCreated = model.Model(thisArchit,
                                   thisLossFunction,
                                   thisOptim,
                                   thisTrainer,
                                   thisEvaluator,
                                   thisDevice,
                                   thisName,
                                   saveDir)

        # Store it
        modelsGNN[thisName] = modelCreated

        # Write the main hyperparameters
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
            
    if doPrint:
        print("Model initialization... COMPLETE")

    #%%##################################################################
    #                                                                   #
    #                    TRAINING                                       #
    #                                                                   #
    #####################################################################
    
    print("")
    
    # We train each model separately
    
    for thisModel in modelsGNN.keys():
        
        if doPrint:
            print("Training model %s..." % thisModel)
        
        # Remember that modelsGNN.keys() has the split numbering as well as the
        # name, while modelList has only the name. So we need to map the
        # specific model for this specific split with the actual model name,
        # since there are several variables that are indexed by the model name
        # (for instance, the training options, or the dictionaries saving the
        # loss values)
        for m in modelList:
            if m in thisModel:
                modelName = m
        
        # Identify the specific split number at training time
        if nDataSplits > 1:
            trainingOptsPerModel[modelName]['graphNo'] = split
        
        # Train the model
        thisTrainVars = modelsGNN[thisModel].train(data,
                                                   nEpochs,
                                                   batchSize,
                                                   **trainingOptsPerModel[modelName])

        if doFigs:
        # Find which model to save the results (when having multiple
        # realizations)
            lossTrain[modelName][split] = thisTrainVars['lossTrain']
            costTrain[modelName][split] = thisTrainVars['costTrain']
            lossValid[modelName][split] = thisTrainVars['lossValid']
            costValid[modelName][split] = thisTrainVars['costValid']
                    
    # And we also need to save 'nBatches' but is the same for all models, so
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

    if doPrint:
        print("Total testing RMSE", end = '', flush = True)
        if nDataSplits > 1:
            print(" (Split %02d)" % split, end = '', flush = True)
        print(":", flush = True)
        

    for thisModel in modelsGNN.keys():
        
        # Same as before, separate the model name from the data split
        # realization number
        for m in modelList:
            if m in thisModel:
                modelName = m

        # Evaluate the model
        thisEvalVars = modelsGNN[thisModel].evaluate(data)
        
        # Save the outputs
        thisCostBest = thisEvalVars['costBest']
        thisCostLast = thisEvalVars['costLast']
        
        # Write values
        writeVarValues(varsFile,
                       {'costBest%s' % thisModel: thisCostBest,
                        'costLast%s' % thisModel: thisCostLast})

        # Now check which is the model being trained
        costBest[modelName][split] = thisCostBest
        costLast[modelName][split] = thisCostLast
        # This is so that we can later compute a total accuracy with
        # the corresponding error.
        
        if doPrint:
            print("\t%s: %.4f [Best] %.4f [Last]" % (thisModel, thisCostBest,
                                                     thisCostLast))

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanCostBest = {} # Mean across data splits
meanCostLast = {} # Mean across data splits
stdDevCostBest = {} # Standard deviation across data splits
stdDevCostLast = {} # Standard deviation across data splits

if doPrint:
    print("\nFinal evaluations (%02d data splits)" % (nDataSplits))

for thisModel in modelList:
    # Convert the lists into a nDataSplits vector
    costBest[thisModel] = np.array(costBest[thisModel])
    costLast[thisModel] = np.array(costLast[thisModel])

    # And now compute the statistics (across graphs)
    meanCostBest[thisModel] = np.mean(costBest[thisModel])
    meanCostLast[thisModel] = np.mean(costLast[thisModel])
    stdDevCostBest[thisModel] = np.std(costBest[thisModel])
    stdDevCostLast[thisModel] = np.std(costLast[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %6.4f (+-%6.4f) [Best] %6.4f (+-%6.4f) [Last]" % (
                thisModel,
                meanCostBest[thisModel],
                stdDevCostBest[thisModel],
                meanCostLast[thisModel],
                stdDevCostLast[thisModel]))

    # Save values
    writeVarValues(varsFile,
               {'meanCostBest%s' % thisModel: meanCostBest[thisModel],
                'stdDevCostBest%s' % thisModel: stdDevCostBest[thisModel],
                'meanCostLast%s' % thisModel: meanCostLast[thisModel],
                'stdDevCostLast%s' % thisModel : stdDevCostLast[thisModel]})
    
# Save the printed info into the .txt file as well
with open(varsFile, 'a+') as file:
    file.write("Final evaluations (%02d data splits)\n" % (nDataSplits))
    for thisModel in modelList:
        file.write("\t%s: %6.4f (+-%6.4f) [Best] %6.4f (+-%6.4f) [Last]\n" % (
                   thisModel,
                   meanCostBest[thisModel],
                   stdDevCostBest[thisModel],
                   meanCostLast[thisModel],
                   stdDevCostLast[thisModel]))
    file.write('\n')

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################

# Finally, we might want to plot several quantities of interest

if doFigs and doSaveVars:

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
    meanCostTrain = {}
    meanLossValid = {}
    meanCostValid = {}
    stdDevLossTrain = {}
    stdDevCostTrain = {}
    stdDevLossValid = {}
    stdDevCostValid = {}
    # Initialize the variables
    for thisModel in modelList:
        # Transform into np.array
        lossTrain[thisModel] = np.array(lossTrain[thisModel])
        costTrain[thisModel] = np.array(costTrain[thisModel])
        lossValid[thisModel] = np.array(lossValid[thisModel])
        costValid[thisModel] = np.array(costValid[thisModel])
        # Each of one of these variables should be of shape
        # nDataSplits x numberOfTrainingSteps
        # And compute the statistics
        meanLossTrain[thisModel] = np.mean(lossTrain[thisModel], axis = 0)
        meanCostTrain[thisModel] = np.mean(costTrain[thisModel], axis = 0)
        meanLossValid[thisModel] = np.mean(lossValid[thisModel], axis = 0)
        meanCostValid[thisModel] = np.mean(costValid[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = np.std(lossTrain[thisModel], axis = 0)
        stdDevCostTrain[thisModel] = np.std(costTrain[thisModel], axis = 0)
        stdDevLossValid[thisModel] = np.std(lossValid[thisModel], axis = 0)
        stdDevCostValid[thisModel] = np.std(costValid[thisModel], axis = 0)

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
    varsPickle['meanCostTrain'] = meanCostTrain
    varsPickle['stdDevCostTrain'] = stdDevCostTrain
    varsPickle['meanLossValid'] = meanLossValid
    varsPickle['stdDevLossValid'] = stdDevLossValid
    varsPickle['meanCostValid'] = meanCostValid
    varsPickle['stdDevCostValid'] = stdDevCostValid
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
            meanCostTrain[thisModel] = meanCostTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevCostTrain[thisModel] = stdDevCostTrain[thisModel]\
                                                        [selectSamplesTrain]
    # And same for the validation, if necessary.
    if xAxisMultiplierValid > 1:
        selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
                                       xAxisMultiplierValid)
        for thisModel in modelList:
            meanLossValid[thisModel] = meanLossValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevLossValid[thisModel] = stdDevLossValid[thisModel]\
                                                        [selectSamplesValid]
            meanCostValid[thisModel] = meanCostValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevCostValid[thisModel] = stdDevCostValid[thisModel]\
                                                        [selectSamplesValid]

    #\\\ LOSS (Training and validation) for EACH MODEL
    for key in meanLossTrain.keys():
        lossFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanLossValid[key], yerr = stdDevLossValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
                        bbox_inches = 'tight')

    #\\\ RMSE (Training and validation) for EACH MODEL
    for key in meanCostTrain.keys():
        costFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanCostTrain[key], yerr = stdDevCostTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanCostValid[key], yerr = stdDevCostValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'RMSE')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        costFig.savefig(os.path.join(saveDirFigs,'cost%s.pdf' % key),
                        bbox_inches = 'tight')

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

    # RMSE (validation) for ALL MODELS
    allCostValidFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanCostValid.keys():
        plt.errorbar(xValid, meanCostValid[key], yerr = stdDevCostValid[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'RMSE')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanCostValid.keys()))
    allCostValidFig.savefig(os.path.join(saveDirFigs,'allCostValid.pdf'),
                    bbox_inches = 'tight')

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