# 2019/04/10~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu

# Test the movie recommendation dataset on several architectures.

# When it runs, it produces the following output:
#   - It trains the specified models and saves the best and the last model
#       parameters of each realization on a directory named 'savedModels'.
#   - It saves a pickle file with the torch random state and the numpy random
#       state for reproducibility.
#   - It saves a text file 'hyperparameters.txt' containing the specific
#       (hyper)parameters that control the run, together with the main (scalar)
#       results obtained.
#   - If desired, logs in tensorboardX the training loss and evaluation measure
#       both of the training set and the validation set. These tensorboardX logs
#       are saved in a logsTB directory.
#   - If desired, saves the vector variables of each realization (training and
#       validation loss and evaluation measure, respectively); this is saved
#       in pickle format. These variables are saved in a trainVars directory.
#   - If desired, plots the training and validation loss and evaluation
#       performance for each of the models, together with the training loss and
#       validation evaluation performance for all models. The summarizing
#       variables used to construct the plots are also saved in pickle format. 
#       These plots (and variables) are in a figs directory.

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
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Modules.architectures as archit
import Modules.model as model
import Modules.train as train
import Modules.loss as loss

#\\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

graphType = 'movie' # Graph type: 'user'-based or 'movie'-based
labelID = [50] # Which node to focus on (either a list or the str 'all')
# When 'movie': [1]: Toy Story, [50]: Star Wars, [258]: Contact,
# [100]: Fargo, [180]: Return of the Jedi, [294]: Liar, liar
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

nDataSplits = 1 # Number of data realizations
# Obs.: The built graph depends on the split between training, validation and
# testing. Therefore, we will run several of these splits and average across
# them, to obtain some result that is more robust to this split.
    
# Given that we build the graph from a training split selected at random, it
# could happen that it is disconnected, or directed, or what not. In other 
# words, we might want to force (by removing nodes) some useful characteristics
# on the graph
keepIsolatedNodes = False # If True keeps isolated nodes
forceUndirected = True # If True forces the graph to be undirected
forceConnected = True # If True returns the largest connected component of the
    # graph as the main graph
kNN = 10 # Number of nearest neighbors

maxDataPoints = None # None to consider all data points

#\\\ Save values:
writeVarValues(varsFile,
               {'labelID': labelID,
                'graphType': graphType,
                'ratioTrain': ratioTrain,
                'ratioValid': ratioValid,
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
trainer = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.005 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.SmoothL1Loss
minRatings = 0 # Discard samples (rows and columns) with less than minRatings 
    # ratings
interpolateRatings = False # Interpolate ratings with nearest-neighbors rule
    # before feeding them into the GNN

#\\\ Overall training options
nEpochs = 40 # Number of epochs
batchSize = 5 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'trainer': trainer,
                'learningRate': learningRate,
                'beta1': beta1,
                'beta2': beta2,
                'lossFunction': lossFunction,
                'minRatings': minRatings,
                'interpolateRatings': interpolateRatings,
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

# If the hyperparameter dictionary is called 'hParams' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N' 
# variable after it has been coded).

# The name of the keys in the hyperparameter dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

modelList = []

#\\\\\\\\\\\\\\\\\\\\\
#\\\ SELECTION GNN \\\
#\\\\\\\\\\\\\\\\\\\\\

if doSelectionGNN:

    #\\\ Basic parameters for all the Selection GNN architectures
    
    hParamsSelGNN = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)
    
    hParamsSelGNN['name'] = 'SelGNN'
    # Chosen architecture
    hParamsSelGNN['archit'] = archit.SelectionGNN
    
    # Graph convolutional parameters
    hParamsSelGNN['dimNodeSignals'] = [1, 64, 32] # Features per layer
    hParamsSelGNN['nFilterTaps'] = [5, 5] # Number of filter taps per layer
    hParamsSelGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsSelGNN['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    hParamsSelGNN['poolingFunction'] = gml.NoPool # Summarizing function
    hParamsSelGNN['nSelectedNodes'] = None # To be determined later on
    hParamsSelGNN['poolingSize'] = [1, 1] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Full MLP readout layer (this layer breaks the locality of the solution)
    hParamsSelGNN['dimLayersMLP'] = [1] # Dimension of the fully connected
        # layers after the GCN layers, we just need to output a single scalar
    # Graph structure
    hParamsSelGNN['GSO'] = None # To be determined later on, based on data
    hParamsSelGNN['order'] = None # Not used because there is no pooling

#\\\\\\\\\\\\
#\\\ MODEL 1: Selection GNN with 1 less layer
#\\\\\\\\\\\\

    hParamsSelGNN1Ly = deepcopy(hParamsSelGNN)

    hParamsSelGNN1Ly['name'] += '1Ly' # Name of the architecture
    
    hParamsSelGNN1Ly['dimNodeSignals'] = hParamsSelGNN['dimNodeSignals'][0:-1]
    hParamsSelGNN1Ly['nFilterTaps'] = hParamsSelGNN['nFilterTaps'][0:-1]
    hParamsSelGNN1Ly['poolingSize'] = hParamsSelGNN['poolingSize'][0:-1]

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsSelGNN1Ly)
    modelList += [hParamsSelGNN1Ly['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 2: Selection GNN with all Layers
#\\\\\\\\\\\\

    hParamsSelGNN2Ly = deepcopy(hParamsSelGNN)

    hParamsSelGNN2Ly['name'] += '2Ly' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsSelGNN2Ly)
    modelList += [hParamsSelGNN2Ly['name']]

    
#\\\\\\\\\\\\\\\\\
#\\\ LOCAL GNN \\\
#\\\\\\\\\\\\\\\\\

if doLocalGNN:

    #\\\ Basic parameters for all the Local GNN architectures
    
    hParamsLclGNN = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)
    
    hParamsLclGNN['name'] = 'LclGNN'
    # Chosen architecture
    hParamsLclGNN['archit'] = archit.LocalGNN
    
    # Graph convolutional parameters
    hParamsLclGNN['dimNodeSignals'] = [1, 64, 32] # Features per layer
    hParamsLclGNN['nFilterTaps'] = [5, 5] # Number of filter taps per layer
    hParamsLclGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsLclGNN['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    hParamsLclGNN['poolingFunction'] = gml.NoPool # Summarizing function
    hParamsLclGNN['nSelectedNodes'] = None # To be determined later on
    hParamsLclGNN['poolingSize'] = [1, 1] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Readout layer: local linear combination of features
    hParamsLclGNN['dimReadout'] = [1] # Dimension of the fully connected layers
        # after the GCN layers (map); this fully connected layer is applied only
        # at each node, without any further exchanges nor considering all nodes
        # at once, making the architecture entirely local.
    # Graph structure
    hParamsLclGNN['GSO'] = None # To be determined later on, based on data
    hParamsLclGNN['order'] = None # Not used because there is no pooling

#\\\\\\\\\\\\
#\\\ MODEL 3: Local GNN with 1 less layer
#\\\\\\\\\\\\

    hParamsLclGNN1Ly = deepcopy(hParamsLclGNN)

    hParamsLclGNN1Ly['name'] += '1Ly' # Name of the architecture
    
    hParamsLclGNN1Ly['dimNodeSignals'] = hParamsLclGNN['dimNodeSignals'][0:-1]
    hParamsLclGNN1Ly['nFilterTaps'] = hParamsLclGNN['nFilterTaps'][0:-1]
    hParamsLclGNN1Ly['poolingSize'] = hParamsLclGNN['poolingSize'][0:-1]

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsLclGNN1Ly)
    modelList += [hParamsLclGNN1Ly['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 4: Local GNN with all Layers
#\\\\\\\\\\\\

    hParamsLclGNN2Ly = deepcopy(hParamsLclGNN)

    hParamsLclGNN2Ly['name'] += '2Ly' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsLclGNN2Ly)
    modelList += [hParamsLclGNN2Ly['name']]

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 0 # After how many training steps, print the partial results
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
    device = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
# Notify:
if doPrint:
    print("Device selected: %s" % device)

#\\\ Logging options
if doLogging:
    # If logging is on, load the tensorboard visualizer and initialize it
    from Utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
accBest = {} # Accuracy for the best model
accLast = {} # Accuracy for the last model
for thisModel in modelList: # Create an element for each split realization,
    accBest[thisModel] = [None] * nDataSplits
    accLast[thisModel] = [None] * nDataSplits


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
    data = Utils.dataTools.MovieLens(graphType, labelID, ratioTrain, ratioValid,
                                     dataDir, keepIsolatedNodes,
                                     forceUndirected, forceConnected, kNN,
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
    # data.to(device)
    # And the corresponding feature dimension that we will need to use
    data.expandDims()
    
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
        hParamsDict = deepcopy(eval('hParams' + thisModel))
        
        # Now, this dictionary has all the hyperparameters that we need to pass
        # to the architecture, but it also has the 'name' and 'archit' that
        # we do not need to pass them. So we are going to get them out of
        # the dictionary
        thisName = hParamsDict.pop('name')
        callArchit = hParamsDict.pop('archit')
        
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
        thisTrainer = trainer
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2

        #\\\ Ordering
        S = G.S.copy()/np.max(np.real(G.E))
        # Do not forget to add the GSO to the input parameters of the archit
        hParamsDict['GSO'] = S
        # Add the number of nodes for the no-pooling part
        if '1Ly' in thisName:
            hParamsDict['nSelectedNodes'] = [nNodes]
        elif '2Ly' in thisName:
            hParamsDict['nSelectedNodes'] = [nNodes, nNodes]
        
        ################
        # ARCHITECTURE #
        ################

        thisArchit = callArchit(**hParamsDict)
        thisArchit.to(device)
        
        #############
        # OPTIMIZER #
        #############

        if thisTrainer == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate,
                                   betas = (beta1, beta2))
        elif thisTrainer == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(),
                                  lr = learningRate)
        elif thisTrainer == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)

        ########
        # LOSS #
        ########

        thisLossFunction = loss.adaptExtraDimensionLoss(lossFunction)

        #########
        # MODEL #
        #########

        modelCreated = model.Model(thisArchit,
                                   thisLossFunction,
                                   thisOptim,
                                   thisName, saveDir)

        modelsGNN[thisName] = modelCreated

        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisTrainer': thisTrainer,
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


    ############
    # TRAINING #
    ############

    # On top of the rest of the training options, we pass the identification
    # of this specific data split realization.

    if nDataSplits > 1:
        trainingOptions['graphNo'] = split

    # This is the function that trains the models detailed in the dictionary
    # modelsGNN using the data data, with the specified training options.
    train.MultipleModels(modelsGNN, data,
                         nEpochs = nEpochs, batchSize = batchSize,
                         **trainingOptions)

    #%%##################################################################
    #                                                                   #
    #                    EVALUATION                                     #
    #                                                                   #
    #####################################################################

    # Now that the model has been trained, we evaluate them on the test
    # samples.

    # We have two versions of each model to evaluate: the one obtained
    # at the best result of the validation step, and the last trained model.

    ########
    # DATA #
    ########

    xTest, yTest = data.getSamples('test')
    xTest = xTest.to(device)
    yTest = yTest.to(device)

    ##############
    # BEST MODEL #
    ##############

    if doPrint:
        print("Total testing RMSE (Best):", flush = True)

    for key in modelsGNN.keys():

        with torch.no_grad():
            # Process the samples
            if 'singleNodeForward' in dir(modelsGNN[key].archit):
                if 'getLabelID' in dir(data):
                    targetIDs = data.getLabelID('test')
                    yHatTest = modelsGNN[key].archit\
                                            .singleNodeForward(xTest, targetIDs)
            else:
                yHatTest = modelsGNN[key].archit(xTest)
            # yHatTest is of shape
            #   testSize x numberOfClasses
            # We compute the accuracy
            thisAccBest = data.evaluate(yHatTest, yTest)

        if doPrint:
            print("%s: %6.4f" % (key, thisAccBest), flush = True)

        # Save value
        writeVarValues(varsFile,
                   {'accBest%s' % key: thisAccBest})

        # Now check which is the model being trained
        for thisModel in modelList:
            # If the name in the modelList is contained in the name with
            # the key, then that's the model, and save it
            # For example, if 'SelGNNDeg' is in thisModelList, then the
            # correct key will read something like 'SelGNNDegG01' so
            # that's the one to save.
            if thisModel in key:
                accBest[thisModel][split] = thisAccBest.item()
            # This is so that we can later compute a total accuracy with
            # the corresponding error.

        del yHatTest
        del thisAccBest

    ##############
    # LAST MODEL #
    ##############

    # And repeat for the last model

    if doPrint:
        print("Total testing RMSE (Last):", flush = True)

    # Update order and adapt dimensions
    for key in modelsGNN.keys():
        modelsGNN[key].load(label = 'Last')

        with torch.no_grad():
            # Process the samples
            if 'singleNodeForward' in dir(modelsGNN[key].archit):
                if 'getLabelID' in dir(data):
                    targetIDs = data.getLabelID('test')
                    yHatTest = modelsGNN[key].archit\
                                     .singleNodeForward(xTest, targetIDs)
            else:
                yHatTest = modelsGNN[key].archit(xTest)
            # yHatTest is of shape
            #   testSize x numberOfClasses
            # We compute the accuracy
            thisAccLast = data.evaluate(yHatTest, yTest)

        if doPrint:
            print("%s: %6.4f" % (key, thisAccLast), flush = True)

        # Save values:
        writeVarValues(varsFile,
                   {'accLast%s' % key: thisAccLast})
        # And repeat for the last model:
        for thisModel in modelList:
            if thisModel in key:
                accLast[thisModel][split] = thisAccLast.item()

        del yHatTest
        del thisAccLast
    
    del yTest
    del xTest

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanAccBest = {} # Mean across data splits
meanAccLast = {} # Mean across data splits
stdDevAccBest = {} # Standard deviation across data splits
stdDevAccLast = {} # Standard deviation across data splits

if doPrint:
    print("\nFinal evaluations (%02d data splits)" % (nDataSplits))

for thisModel in modelList:
    # Convert the lists into a nDataSplits vector
    accBest[thisModel] = np.array(accBest[thisModel])
    accLast[thisModel] = np.array(accLast[thisModel])

    # And now compute the statistics (across graphs)
    meanAccBest[thisModel] = np.mean(accBest[thisModel])
    meanAccLast[thisModel] = np.mean(accLast[thisModel])
    stdDevAccBest[thisModel] = np.std(accBest[thisModel])
    stdDevAccLast[thisModel] = np.std(accLast[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %6.4f (+-%6.4f) [Best] %6.4f (+-%6.4f) [Last]" % (
                thisModel,
                meanAccBest[thisModel],
                stdDevAccBest[thisModel],
                meanAccLast[thisModel],
                stdDevAccLast[thisModel]))

    # Save values
    writeVarValues(varsFile,
               {'meanAccBest%s' % thisModel: meanAccBest[thisModel],
                'stdDevAccBest%s' % thisModel: stdDevAccBest[thisModel],
                'meanAccLast%s' % thisModel: meanAccLast[thisModel],
                'stdDevAccLast%s' % thisModel : stdDevAccLast[thisModel]})

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

    # Again, we have training and validation metrics (loss and accuracy
    # -evaluation-) for many runs, so we need to carefully load them and compute
    # the relevant statistics from these realizations.

    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    evalTrain = {}
    lossValid = {}
    evalValid = {}
    # Initialize the splits dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nDataSplits
        evalTrain[thisModel] = [None] * nDataSplits
        lossValid[thisModel] = [None] * nDataSplits
        evalValid[thisModel] = [None] * nDataSplits

    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)

    #\\\ LOAD DATA:
    # Path where the saved training variables should be
    pathToTrainVars = os.path.join(saveDir,'trainVars')
    # Get all the training files:
    allTrainFiles = next(os.walk(pathToTrainVars))[2]
    # Go over each of them (this can't be empty since we are also checking for
    # doSaveVars to be true, what guarantees that the variables have been
    # saved.)
    for file in allTrainFiles:
        # Check that it is a pickle file
        if '.pkl' in file:
            # Open the file
            with open(os.path.join(pathToTrainVars,file),'rb') as fileTrainVars:
                # Load it
                thisVarsDict = pickle.load(fileTrainVars)
                # store them
                nBatches = thisVarsDict['nBatches']
                thisLossTrain = thisVarsDict['lossTrain']
                thisEvalTrain = thisVarsDict['evalTrain']
                thisLossValid = thisVarsDict['lossValid']
                thisEvalValid = thisVarsDict['evalValid']
                # This graph is, actually, the data split dimension
                if 'graphNo' in thisVarsDict.keys():
                    thisG = thisVarsDict['graphNo']
                else:
                    thisG = 0
                # And add them to the corresponding variables
                for key in thisLossTrain.keys():
                # This part matches each data realization (matched through
                # the graphNo key) with each specific model.
                    for thisModel in modelList:
                        if thisModel in key:
                            lossTrain[thisModel][thisG] = thisLossTrain[key]
                            evalTrain[thisModel][thisG] = thisEvalTrain[key]
                            lossValid[thisModel][thisG] = thisLossValid[key]
                            evalValid[thisModel][thisG] = thisEvalValid[key]
    # Now that we have collected all the results, we have that each of the four
    # variables (lossTrain, evalTrain, lossValid, evalValid) has a list for
    # each key in the dictionary. This list goes through the data split.
    # Each split realization is actually an np.array.

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    meanLossTrain = {}
    meanEvalTrain = {}
    meanLossValid = {}
    meanEvalValid = {}
    stdDevLossTrain = {}
    stdDevEvalTrain = {}
    stdDevLossValid = {}
    stdDevEvalValid = {}
    # Initialize the variables
    for thisModel in modelList:
        # Transform into np.array
        lossTrain[thisModel] = np.array(lossTrain[thisModel])
        evalTrain[thisModel] = np.array(evalTrain[thisModel])
        lossValid[thisModel] = np.array(lossValid[thisModel])
        evalValid[thisModel] = np.array(evalValid[thisModel])
        # Each of one of these variables should be of shape
        # nDataSplits x numberOfTrainingSteps
        # And compute the statistics
        meanLossTrain[thisModel] = np.mean(lossTrain[thisModel], axis = 0)
        meanEvalTrain[thisModel] = np.mean(evalTrain[thisModel], axis = 0)
        meanLossValid[thisModel] = np.mean(lossValid[thisModel], axis = 0)
        meanEvalValid[thisModel] = np.mean(evalValid[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = np.std(lossTrain[thisModel], axis = 0)
        stdDevEvalTrain[thisModel] = np.std(evalTrain[thisModel], axis = 0)
        stdDevLossValid[thisModel] = np.std(lossValid[thisModel], axis = 0)
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
    varsPickle['meanEvalTrain'] = meanEvalTrain
    varsPickle['stdDevEvalTrain'] = stdDevEvalTrain
    varsPickle['meanLossValid'] = meanLossValid
    varsPickle['stdDevLossValid'] = stdDevLossValid
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
            meanEvalTrain[thisModel] = meanEvalTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevEvalTrain[thisModel] = stdDevEvalTrain[thisModel]\
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
    for key in meanEvalTrain.keys():
        accFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanEvalTrain[key], yerr = stdDevEvalTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'RMSE')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        accFig.savefig(os.path.join(saveDirFigs,'eval%s.pdf' % key),
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
    allEvalValid = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanEvalValid.keys():
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'RMSE')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanEvalValid.keys()))
    allEvalValid.savefig(os.path.join(saveDirFigs,'allEvalValid.pdf'),
                    bbox_inches = 'tight')
