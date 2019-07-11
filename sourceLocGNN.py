# 2018/12/03~2018/07/12
# Fernando Gama, fgama@seas.upenn.edu

# In this code, we simulate the source localization problem, and test the 
# following architectures
#   - EdgeVariantGNN (1 layer, both hybrid and full)
#   - NodeVariantGNN (1 layer, both hybrid and full)
#   - GraphAttentionNetwork (1 layer)

# When it runs, it produces the following output:
#   - It trains the specified models and saves the best and the last model
#       of each realization on a directory named 'savedModels'
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
#       both in pickle and in Matlab(R) format. These variables are saved in a
#       trainVars directory.
#   - If desired, plots the training and validation loss and evaluation
#       performance for each of the models, together with the training loss and
#       validation evaluation performance for all models. The summarizing
#       variables used to construct the plots are also saved in both pickle and
#       Matlab(R) format. These plots (and variables) are in a figs directory.

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
from scipy.io import savemat
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

#\\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

graphType = 'SBM' # Type of graph: 'SBM', 'FacebookEgo', 'SmallWorld'

thisFilename = 'sourceLocGNN' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run
if graphType == 'FacebookEgo':
    dataDir = os.path.join('datasets','facebookEgo')

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + graphType + '-' + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters are results will be saved.
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

nTrain = 1000 # Number of training samples
nValid = int(0.24 * nTrain) # Number of validation samples
nTest = 200 # Number of testing samples
tMax = None # Maximum number of diffusion times (A^t for t < tMax)

nDataRealizations = 10 # Number of data realizations
nGraphRealizations = 10 # Number of graph realizations
nClasses = 2 # Number of source nodes to select

nNodes = 40 # Number of nodes
graphOptions = {} # Dictionary of options to pass to the createGraph function
if graphType == 'SBM':
    graphOptions['nCommunities'] = nClasses # Number of communities
    graphOptions['probIntra'] = 0.8 # Intracommunity probability
    graphOptions['probInter'] = 0.2 # Intercommunity probability
elif graphType == 'SmallWorld':
    graphOptions['probEdge'] = 0.5 # Edge probability
    graphOptions['probRewiring'] = 0.1 # Probability of rewiring
elif graphType == 'FacebookEgo':
    graphOptions['isolatedNodes'] = False # If True keeps isolated nodes
    graphOptions['forceConnected'] = True # If True removes nodes (from lowest to highest degree)
        # until the resulting graph is connected.
    use234 = True # Use a smaller 234-matrix with 2-communities instead of the full
        # graph with around 4k users
    nGraphRealizations = 1 # Number of graph realizations
    if use234:
        nClasses = 2

#\\\ Save values:
writeVarValues(varsFile, {'nNodes': nNodes, 'graphType': graphType})
writeVarValues(varsFile, graphOptions)
writeVarValues(varsFile, {'nTrain': nTest,
                          'nValid': nValid,
                          'nTest': nTest,
                          'tMax': tMax,
                          'nDataRealizations':nDataRealizations,
                          'nGraphRealizations': nGraphRealizations,
                          'nClasses': nClasses,
                          'useGPU': useGPU})

############
# TRAINING #
############

#\\\ Individual model training options
trainer = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.001 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.CrossEntropyLoss() # This applies a softmax before feeding
    # it into the NLL, so we don't have to apply the softmax ourselves.

#\\\ Overall training options
nEpochs = 80 # Number of epochs
batchSize = 20 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 20 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'trainer': trainer,
                'learningRate': learningRate,
                'beta1': beta1,
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

# Select desired architectures
doGAT = True
doNodeVariantGNN = True
doEdgeVariantGNN = True

# Select type of GNN
doHybrid = True
doFull = True

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. That is, any new architecture in this part, needs also
# to be coded later on. This is just to be easy to change the parameters once
# the architecture is created. Do not forget to add the name of the architecture
# to modelList.

modelList = []

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#\\\ GRAPH ATTENTION NETWORK \\\
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

#\\\\\\\\\\\\
#\\\ MODEL 1: Graph Attention Network
#\\\\\\\\\\\\

if doGAT:

    hParamsGrpAttNet = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsGrpAttNet['name'] = 'GrpAttNet' # Name of the architecture

    #\\\ Architecture parameters
    hParamsGrpAttNet['F'] = [1, 16] # Features per layer
    hParamsGrpAttNet['K'] = [4] # Number of attention heads
    hParamsGrpAttNet['sigma'] = nn.functional.relu # Selected nonlinearity
    hParamsGrpAttNet['rho'] = gml.NoPool # Summarizing function
    hParamsGrpAttNet['alpha'] = [1] # alpha-hop neighborhood that is
        #affected by the summary
    hParamsGrpAttNet['dimLayersMLP'] = [nClasses] # Dimension of the fully
        # connected layers after the GCN layers
    hParamsGrpAttNet['bias'] = True # Decide whether to include a bias term

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGrpAttNet)
    modelList += [hParamsGrpAttNet['name']]
    
#\\\\\\\\\\\\\\\\\\\\\\\\
#\\\ NODE-VARIANT GNN \\\
#\\\\\\\\\\\\\\\\\\\\\\\\
    
# Common parameters for both node-variants architectures

if doNodeVariantGNN:

    hParamsNdVGNN = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    #\\\ Architecture parameters
    hParamsNdVGNN['F'] = [1, 16] # Features per layer
    hParamsNdVGNN['K'] = [4] # Number of shift taps per layer
    hParamsNdVGNN['M'] = [4] # Number of node taps per layer
    hParamsNdVGNN['bias'] = True # Decide whether to include a bias term
    hParamsNdVGNN['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsNdVGNN['rho'] = gml.NoPool # Summarizing function
    hParamsNdVGNN['alpha'] = [1] # alpha-hop neighborhood that is
        #affected by the summary
    hParamsNdVGNN['dimLayersMLP'] = [nClasses] # Dimension of the fully
        # connected layers after the GCN layers
        
#\\\\\\\\\\\\
#\\\ MODEL 2: Hybrid Node-Variant GNN with selected nodes determined by degree
#\\\\\\\\\\\\

if doNodeVariantGNN and doHybrid:

    hParamsNdVGNNhbr = deepcopy(hParamsNdVGNN)

    hParamsNdVGNNhbr['name'] = 'NdVGNNhbr' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsNdVGNNhbr)
    modelList += [hParamsNdVGNNhbr['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 3: Full Node-Variant GNN
#\\\\\\\\\\\\

if doNodeVariantGNN and doFull:

    hParamsNdVGNNfll = deepcopy(hParamsNdVGNN)

    hParamsNdVGNNfll['name'] = 'NdVGNNfll' # Name of the architecture
    
    # We need to change hParamsNdVGNNfll['M'] to be equal to the number of 
    # nodes, but we do not know yet how many nodes there will be.

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsNdVGNNfll)
    modelList += [hParamsNdVGNNfll['name']]
    
#\\\\\\\\\\\\\\\\\\\\\\\\
#\\\ EDGE-VARIANT GNN \\\
#\\\\\\\\\\\\\\\\\\\\\\\\
    
# Common parameters for both node-variants architectures

if doEdgeVariantGNN:

    hParamsEdVGNN = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    #\\\ Architecture parameters
    hParamsEdVGNN['F'] = [1, 16] # Features per layer
    hParamsEdVGNN['K'] = [4] # Number of shift taps per layer
    hParamsEdVGNN['M'] = [4] # Number of node taps per layer
    hParamsEdVGNN['bias'] = True # Decide whether to include a bias term
    hParamsEdVGNN['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsEdVGNN['rho'] = gml.NoPool # Summarizing function
    hParamsEdVGNN['alpha'] = [1] # alpha-hop neighborhood that is
        #affected by the summary
    hParamsEdVGNN['dimLayersMLP'] = [nClasses] # Dimension of the fully
        # connected layers after the GCN layers
        
#\\\\\\\\\\\\
#\\\ MODEL 4: Hybrid Edge-Variant GNN with selected nodes determined by degree
#\\\\\\\\\\\\

if doEdgeVariantGNN and doHybrid:

    hParamsEdVGNNhbr = deepcopy(hParamsEdVGNN)

    hParamsEdVGNNhbr['name'] = 'EdVGNNhbr' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsEdVGNNhbr)
    modelList += [hParamsEdVGNNhbr['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 5: Full Edge-Variant GNN
#\\\\\\\\\\\\

if doEdgeVariantGNN and doFull:

    hParamsEdVGNNfll = deepcopy(hParamsEdVGNN)

    hParamsEdVGNNfll['name'] = 'EdVGNNfll' # Name of the architecture
    
    # We need to change hParamsEdVGNNfll['M'] to be equal to the number of 
    # nodes, but we do not know yet how many nodes there will be.

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsEdVGNNfll)
    modelList += [hParamsEdVGNNfll['name']]

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
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 10 # How many validation steps in between those shown,
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
    from Utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each for the trained models.
# It basically is a dictionary, containing a list of lists. The key of the
# dictionary determines de the model, then the first list index determines
# which graph, and the second list index, determines which realization within
# that graph. Then, this will be converted to numpy to compute mean and standard
# deviation (across the graph dimension).
accBest = {} # Accuracy for the best model
accLast = {} # Accuracy for the last model
for thisModel in modelList: # Create an element for each graph realization,
    # each of these elements will later be another list for each realization.
    # That second list is created empty and just appends the results.
    accBest[thisModel] = [None] * nGraphRealizations
    accLast[thisModel] = [None] * nGraphRealizations

####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of this options was decided above with the rest of the parameters.
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
#                    GRAPH REALIZATION                              #
#                                                                   #
#####################################################################

# Start generating a new graph for each of the number of graph realizations that
# we previously specified.

# Unless it's the Facebook graph, which is fixed

# Load the graph and select the source nodes

if graphType == 'FacebookEgo':

    #########
    # GRAPH #
    #########

    if doPrint:
        print("Load data...", flush = True, end = ' ')

    # Create graph
    facebookData = Utils.dataTools.FacebookEgo(dataDir, use234)
    adjacencyMatrix = facebookData.getAdjacencyMatrix(use234)
    assert adjacencyMatrix is not None
    nNodes = adjacencyMatrix.shape[0]

    if doPrint:
        print("OK")
    # Now, to create the proper graph object, since we're going to use
    # 'fuseEdges' option in createGraph, we are going to add an extra dimension
    # to the adjacencyMatrix (to indicate there's only one matrix in the 
    # collection that we should be fusing)
    adjacencyMatrix = adjacencyMatrix.reshape([1, nNodes, nNodes])
    nodeList = []
    extraComponents = []
    if doPrint:
        print("Creating graph...", flush = True, end = ' ')
    graphOptions['adjacencyMatrices'] = adjacencyMatrix
    graphOptions['nodeList'] = nodeList
    graphOptions['extraComponents'] = extraComponents
    graphOptions['aggregationType'] = 'sum'
    graphOptions['normalizationType'] = 'no'
    graphOptions['forceUndirected'] = True
    
    G = graphTools.Graph('fuseEdges', nNodes, graphOptions)
    G.computeGFT() # Compute the eigendecomposition of the stored GSO
    
    nNodes = G.N
    
    if doPrint:
        print("OK")
    
    ################
    # SOURCE NODES #
    ################

    if doPrint:
        print("Selecting source nodes...", end = ' ', flush = True)
    # For the source localization problem, we have to select which ones, of all
    # the nodes, will act as source nodes. This is determined by a list of
    # indices indicating which nodes to choose as sources.
    sourceNodes = graphTools.computeSourceNodes(G.A, nClasses)
    if use234:
        sourceNodes = [38, 224]
    
    #\\\ Save values:
    writeVarValues(varsFile,
                   {'sourceNodes': sourceNodes})
    
    if doPrint:
        print("OK")
        
# These are the architectures with layers that depend on the number of nodes
# (like, when there is no pooling), and since the number of nodes could have
# changed, if we selected the FB graph, this is the moment to add those values

if doGAT:
    hParamsGrpAttNet['N'] = [nNodes]
if doNodeVariantGNN and doHybrid:
    hParamsNdVGNNhbr['N'] = [nNodes]
if doNodeVariantGNN and doFull:
    hParamsNdVGNNfll['N'] = [nNodes]
    hParamsNdVGNNfll['M'] = [nNodes]
if doEdgeVariantGNN and doHybrid:
    hParamsEdVGNNhbr['N'] = [nNodes]
if doEdgeVariantGNN and doFull:
    hParamsEdVGNNfll['N'] = [nNodes]
    hParamsEdVGNNfll['M'] = [nNodes]

for graph in range(nGraphRealizations):

    # The accBest and accLast variables, for each model, have a list with a
    # total number of elements equal to the number of graphs we will generate
    # Now, for each graph, we have multiple data realization, so we want, for
    # each graph, to create a list to hold each of those values
    for thisModel in modelList:
        accBest[thisModel][graph] = []
        accLast[thisModel][graph] = []

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    if graphType != 'FacebookEgo':
        # If the graph type is the Facebook one, then that graph is fixed,
        # so we don't have to keep changing it.

        #########
        # GRAPH #
        #########

        # Create graph
        G = graphTools.Graph(graphType, nNodes, graphOptions)
        G.computeGFT() # Compute the eigendecomposition of the stored GSO

        ################
        # SOURCE NODES #
        ################

        # For the source localization problem, we have to select which ones, of
        # all the nodes, will act as source nodes. This is determined by a list
        # of indices indicating which nodes to choose as sources.
        sourceNodes = graphTools.computeSourceNodes(G.A, nClasses)
        
        #\\\ Save values:
        writeVarValues(varsFile,
                       {'sourceNodes': sourceNodes})

    # We have now created the graph and selected the source nodes on that graph.
    # So now we proceed to generate random data realizations, different
    # realizations of diffusion processes.

    for realization in range(nDataRealizations):

        ############
        # DATASETS #
        ############

        #   Now that we have the list of nodes we are using as sources, then we
        #   can go ahead and generate the datasets.
        data = Utils.dataTools.SourceLocalization(G, nTrain, nValid, nTest,
                                                  sourceNodes, tMax = tMax)
        data.astype(torch.float64)
        data.to(device)

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

        #%%\\\\\\\\\\
        #\\\ MODEL 1: Graph Attention Network
        #\\\\\\\\\\\\
    
        if doGAT:
    
            thisName = hParamsGrpAttNet['name']
    
            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
                
            if doPrint:
                print("\tInitializing %s..." % thisName, end = ' ',flush = True)
    
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
            S, order = graphTools.permIdentity(G.S/np.max(np.real(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.GraphAttentionNetwork(# Graph attentional layers
                                               hParamsGrpAttNet['F'],
                                               hParamsGrpAttNet['K'],
                                               # Nonlinearity
                                               hParamsGrpAttNet['sigma'],
                                               # Pooling
                                               hParamsGrpAttNet['N'],
                                               hParamsGrpAttNet['rho'],
                                               hParamsGrpAttNet['alpha'],
                                               # MLP
                                               hParamsGrpAttNet['dimLayersMLP'],
                                               hParamsGrpAttNet['bias'],
                                               # Structure
                                               S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr = learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction
    
            #########
            # MODEL #
            #########
    
            GrpAttNet = model.Model(thisArchit, thisLossFunction, thisOptim,
                                 thisName, saveDir, order)
    
            modelsGNN[thisName] = GrpAttNet
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
            if doPrint:
                print("OK")
    
        #%%\\\\\\\\\\
        #\\\ MODEL 2: Hybrid Node-Variant GNN with selected nodes determined by deg
        #\\\\\\\\\\\\
    
        if doNodeVariantGNN and doHybrid:
    
            thisName = hParamsNdVGNNhbr['name']
    
            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
                
            if doPrint:
                print("\tInitializing %s..." % thisName, end = ' ',flush = True)
    
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
            S, order = graphTools.permDegree(G.S/np.max(np.real(G.E)))
            # order is an np.array with the ordering of the nodes with respect to
            # the original GSO (the original GSO is kept in G.S).
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.NodeVariantGNN(# Graph filtering
                                             hParamsNdVGNNhbr['F'],
                                             hParamsNdVGNNhbr['K'],
                                             hParamsNdVGNNhbr['M'],
                                             hParamsNdVGNNhbr['bias'],
                                             # Nonlinearity
                                             hParamsNdVGNNhbr['sigma'],
                                             # Pooling
                                             hParamsNdVGNNhbr['N'],
                                             hParamsNdVGNNhbr['rho'],
                                             hParamsNdVGNNhbr['alpha'],
                                             # MLP
                                             hParamsNdVGNNhbr['dimLayersMLP'],
                                             # Structure
                                             S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction
    
            #########
            # MODEL #
            #########
    
            NdVGNNhbr = model.Model(thisArchit, thisLossFunction, thisOptim,
                                 thisName, saveDir, order)
    
            modelsGNN[thisName] = NdVGNNhbr
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
            if doPrint:
                print("OK")
        
        #%%\\\\\\\\\\
        #\\\ MODEL 3: Full Node-Variant GNN
        #\\\\\\\\\\\\
    
        if doNodeVariantGNN and doFull:
    
            thisName = hParamsNdVGNNfll['name']
    
            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
                
            if doPrint:
                print("\tInitializing %s..." % thisName, end = ' ',flush = True)
    
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
            S, order = graphTools.permIdentity(G.S/np.max(np.real(G.E)))
            # order is an np.array with the ordering of the nodes with respect to
            # the original GSO (the original GSO is kept in G.S).
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.NodeVariantGNN(# Graph filtering
                                             hParamsNdVGNNfll['F'],
                                             hParamsNdVGNNfll['K'],
                                             hParamsNdVGNNfll['M'],
                                             hParamsNdVGNNfll['bias'],
                                             # Nonlinearity
                                             hParamsNdVGNNfll['sigma'],
                                             # Pooling
                                             hParamsNdVGNNfll['N'],
                                             hParamsNdVGNNfll['rho'],
                                             hParamsNdVGNNfll['alpha'],
                                             # MLP
                                             hParamsNdVGNNfll['dimLayersMLP'],
                                             # Structure
                                             S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction
    
            #########
            # MODEL #
            #########
    
            NdVGNNfll = model.Model(thisArchit, thisLossFunction, thisOptim,
                                 thisName, saveDir, order)
    
            modelsGNN[thisName] = NdVGNNfll
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
            if doPrint:
                print("OK")
    
        #%%\\\\\\\\\\
        #\\\ MODEL 4: Hybrid Edge-Variant GNN with selected nodes determined by deg
        #\\\\\\\\\\\\
    
        if doEdgeVariantGNN and doHybrid:
    
            thisName = hParamsEdVGNNhbr['name']
    
            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
                
            if doPrint:
                print("\tInitializing %s..." % thisName, end = ' ',flush = True)
    
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
            S, order = graphTools.permDegree(G.S/np.max(np.real(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.EdgeVariantGNN(# Graph filtering
                                               hParamsEdVGNNhbr['F'],
                                               hParamsEdVGNNhbr['K'],
                                               hParamsEdVGNNhbr['M'],
                                               hParamsEdVGNNhbr['bias'],
                                               # Nonlinearity
                                               hParamsEdVGNNhbr['sigma'],
                                               # Pooling
                                               hParamsEdVGNNhbr['N'],
                                               hParamsEdVGNNhbr['rho'],
                                               hParamsEdVGNNhbr['alpha'],
                                               # MLP
                                               hParamsEdVGNNhbr['dimLayersMLP'],
                                               # Structure
                                               S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction
    
            #########
            # MODEL #
            #########
    
            EdVGNNhbr = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)
    
            modelsGNN[thisName] = EdVGNNhbr
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
            if doPrint:
                print("OK")
        
        #%%\\\\\\\\\\
        #\\\ MODEL 5: Full Edge-Variant GNN
        #\\\\\\\\\\\\
    
        if doEdgeVariantGNN and doFull:
    
            thisName = hParamsEdVGNNfll['name']
    
            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
                
            if doPrint:
                print("\tInitializing %s..." % thisName, end = ' ',flush = True)
    
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
            S, order = graphTools.permIdentity(G.S/np.max(np.real(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.EdgeVariantGNN(# Graph filtering
                                               hParamsEdVGNNfll['F'],
                                               hParamsEdVGNNfll['K'],
                                               hParamsEdVGNNfll['M'],
                                               hParamsEdVGNNfll['bias'],
                                               # Nonlinearity
                                               hParamsEdVGNNfll['sigma'],
                                               # Pooling
                                               hParamsEdVGNNfll['N'],
                                               hParamsEdVGNNfll['rho'],
                                               hParamsEdVGNNfll['alpha'],
                                               # MLP
                                               hParamsEdVGNNfll['dimLayersMLP'],
                                               # Structure
                                               S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction
    
            #########
            # MODEL #
            #########
    
            EdVGNNfll = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)
    
            modelsGNN[thisName] = EdVGNNfll
    
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
        # of this specific graph/data realization.

        if nGraphRealizations > 1:
            trainingOptions['graphNo'] = graph
        if nDataRealizations > 1:
            trainingOptions['realizationNo'] = realization

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

        ##############
        # BEST MODEL #
        ##############

        if doPrint:
            print("Total testing accuracy (Best):", flush = True)

        for key in modelsGNN.keys():
            # Update order and adapt dimensions (this data has one input feature,
            # so we need to add that dimension; make it from B x N to B x F x N)
            xTestOrdered = xTest[:,modelsGNN[key].order].unsqueeze(1)

            with torch.no_grad():
                # Process the samples
                yHatTest = modelsGNN[key].archit(xTestOrdered)
                # yHatTest is of shape
                #   testSize x numberOfClasses
                # We compute the accuracy
                thisAccBest = data.evaluate(yHatTest, yTest)

            if doPrint:
                print("%s: %4.2f%%" % (key, thisAccBest * 100.), flush = True)

            # Save value
            writeVarValues(varsFile,
                       {'accBest%s' % key: thisAccBest})

            # Now check which is the model being trained
            for thisModel in modelList:
                # If the name in the modelList is contained in the name with
                # the key, then that's the model, and save it
                # For example, if 'SelGNNDeg' is in thisModelList, then the
                # correct key will read something like 'SelGNNDegG01R00' so
                # that's the one to save.
                if thisModel in key:
                    accBest[thisModel][graph] += [thisAccBest.item()]
                # This is so that we can later compute a total accuracy with
                # the corresponding error.

        ##############
        # LAST MODEL #
        ##############

        # And repeat for the last model

        if doPrint:
            print("Total testing accuracy (Last):", flush = True)

        # Update order and adapt dimensions
        for key in modelsGNN.keys():
            modelsGNN[key].load(label = 'Last')
            xTestOrdered = xTest[:,modelsGNN[key].order].unsqueeze(1)

            with torch.no_grad():
                # Process the samples
                yHatTest = modelsGNN[key].archit(xTestOrdered)
                # yHatTest is of shape
                #   testSize x numberOfClasses
                # We compute the accuracy
                thisAccLast = data.evaluate(yHatTest, yTest)

            if doPrint:
                print("%s: %4.2f%%" % (key, thisAccLast * 100), flush = True)

            # Save values:
            writeVarValues(varsFile,
                       {'accLast%s' % key: thisAccLast})
            # And repeat for the last model:
            for thisModel in modelList:
                if thisModel in key:
                    accLast[thisModel][graph] += [thisAccLast.item()]

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanAccBestPerGraph = {} # Compute the mean accuracy (best) across all
    # realizations data realizations of a graph
meanAccLastPerGraph = {} # Compute the mean accuracy (last) across all
    # realizations data realizations of a graph
meanAccBest = {} # Mean across graphs (after having averaged across data
    # realizations)
meanAccLast = {} # Mean across graphs
stdDevAccBest = {} # Standard deviation across graphs
stdDevAccLast = {} # Standard deviation across graphs

if doPrint:
    print("\nFinal evaluations (%02d graphs, %02d realizations)" % (
            nGraphRealizations, nDataRealizations))

for thisModel in modelList:
    # Convert the lists into a nGraphRealizations x nDataRealizations matrix
    accBest[thisModel] = np.array(accBest[thisModel])
    accLast[thisModel] = np.array(accLast[thisModel])

    # Compute the mean (across realizations for a given graph)
    meanAccBestPerGraph[thisModel] = np.mean(accBest[thisModel], axis = 1)
    meanAccLastPerGraph[thisModel] = np.mean(accLast[thisModel], axis = 1)

    # And now compute the statistics (across graphs)
    meanAccBest[thisModel] = np.mean(meanAccBestPerGraph[thisModel])
    meanAccLast[thisModel] = np.mean(meanAccLastPerGraph[thisModel])
    stdDevAccBest[thisModel] = np.std(meanAccBestPerGraph[thisModel])
    stdDevAccLast[thisModel] = np.std(meanAccLastPerGraph[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %6.2f%% (+-%6.2f%%) [Best] %6.2f%% (+-%6.2f%%) [Last]" % (
                thisModel,
                meanAccBest[thisModel] * 100,
                stdDevAccBest[thisModel] * 100,
                meanAccLast[thisModel] * 100,
                stdDevAccLast[thisModel] * 100))

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
    # of lists, one list for each graph, and one list for each data realization.
    # Each data realization, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    evalTrain = {}
    lossValid = {}
    evalValid = {}
    # Initialize the graph dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nGraphRealizations
        evalTrain[thisModel] = [None] * nGraphRealizations
        lossValid[thisModel] = [None] * nGraphRealizations
        evalValid[thisModel] = [None] * nGraphRealizations
        # Initialize the data realization dimension with empty lists to then
        # append each realization when we load it.
        for G in range(nGraphRealizations):
            lossTrain[thisModel][G] = []
            evalTrain[thisModel][G] = []
            lossValid[thisModel][G] = []
            evalValid[thisModel][G] = []

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
    # doSaveVars to be true, what guarantees that the variables have been saved.
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
                if 'graphNo' in thisVarsDict.keys():
                    thisG = thisVarsDict['graphNo']
                else:
                    thisG = 0
                if 'realizationNo' in thisVarsDict.keys():
                    thisR = thisVarsDict['realizationNo']
                else:
                    thisR = 0
                # And add them to the corresponding variables
                for key in thisLossTrain.keys():
                # This part matches each realization (saved with a different
                # name due to identification of graph and data realization) with
                # the specific model.
                    for thisModel in modelList:
                        if thisModel in key:
                            lossTrain[thisModel][thisG] += [thisLossTrain[key]]
                            evalTrain[thisModel][thisG] += [thisEvalTrain[key]]
                            lossValid[thisModel][thisG] += [thisLossValid[key]]
                            evalValid[thisModel][thisG] += [thisEvalValid[key]]
    # Now that we have collected all the results, we have that each of the four
    # variables (lossTrain, evalTrain, lossValid, evalValid) has a list of lists
    # for each key in the dictionary. The first list goes through the graph, and
    # for each graph, it goes through data realizations. Each data realization
    # is actually an np.array.

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    meanLossTrainPerGraph = {}
    meanEvalTrainPerGraph = {}
    meanLossValidPerGraph = {}
    meanEvalValidPerGraph = {}
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
        meanLossTrainPerGraph[thisModel] = [None] * nGraphRealizations
        meanEvalTrainPerGraph[thisModel] = [None] * nGraphRealizations
        meanLossValidPerGraph[thisModel] = [None] * nGraphRealizations
        meanEvalValidPerGraph[thisModel] = [None] * nGraphRealizations
        for G in range(nGraphRealizations):
            # Transform into np.array
            lossTrain[thisModel][G] = np.array(lossTrain[thisModel][G])
            evalTrain[thisModel][G] = np.array(evalTrain[thisModel][G])
            lossValid[thisModel][G] = np.array(lossValid[thisModel][G])
            evalValid[thisModel][G] = np.array(evalValid[thisModel][G])
            # So, finally, for each model and each graph, we have a np.array of
            # shape:  nDataRealizations x number_of_training_steps
            # And we have to average these to get the mean across all data
            # realizations for each graph
            meanLossTrainPerGraph[thisModel][G] = \
                                    np.mean(lossTrain[thisModel][G], axis = 0)
            meanEvalTrainPerGraph[thisModel][G] = \
                                    np.mean(evalTrain[thisModel][G], axis = 0)
            meanLossValidPerGraph[thisModel][G] = \
                                    np.mean(lossValid[thisModel][G], axis = 0)
            meanEvalValidPerGraph[thisModel][G] = \
                                    np.mean(evalValid[thisModel][G], axis = 0)
        # And then convert this into np.array for all graphs
        meanLossTrainPerGraph[thisModel] = \
                                    np.array(meanLossTrainPerGraph[thisModel])
        meanEvalTrainPerGraph[thisModel] = \
                                    np.array(meanEvalTrainPerGraph[thisModel])
        meanLossValidPerGraph[thisModel] = \
                                    np.array(meanLossValidPerGraph[thisModel])
        meanEvalValidPerGraph[thisModel] = \
                                    np.array(meanEvalValidPerGraph[thisModel])
        # And compute the statistics
        meanLossTrain[thisModel] = \
                            np.mean(meanLossTrainPerGraph[thisModel], axis = 0)
        meanEvalTrain[thisModel] = \
                            np.mean(meanEvalTrainPerGraph[thisModel], axis = 0)
        meanLossValid[thisModel] = \
                            np.mean(meanLossValidPerGraph[thisModel], axis = 0)
        meanEvalValid[thisModel] = \
                            np.mean(meanEvalValidPerGraph[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = \
                            np.std(meanLossTrainPerGraph[thisModel], axis = 0)
        stdDevEvalTrain[thisModel] = \
                            np.std(meanEvalTrainPerGraph[thisModel], axis = 0)
        stdDevLossValid[thisModel] = \
                            np.std(meanLossValidPerGraph[thisModel], axis = 0)
        stdDevEvalValid[thisModel] = \
                            np.std(meanEvalValidPerGraph[thisModel], axis = 0)

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
    #   Matlab, second:
    varsMatlab = {}
    varsMatlab['nEpochs'] = nEpochs
    varsMatlab['nBatches'] = nBatches
    for thisModel in modelList:
        varsMatlab['meanLossTrain' + thisModel] = meanLossTrain[thisModel]
        varsMatlab['stdDevLossTrain' + thisModel] = stdDevLossTrain[thisModel]
        varsMatlab['meanEvalTrain' + thisModel] = meanEvalTrain[thisModel]
        varsMatlab['stdDevEvalTrain' + thisModel] = stdDevEvalTrain[thisModel]
        varsMatlab['meanLossValid' + thisModel] = meanLossValid[thisModel]
        varsMatlab['stdDevLossValid' + thisModel] = stdDevLossValid[thisModel]
        varsMatlab['meanEvalValid' + thisModel] = meanEvalValid[thisModel]
        varsMatlab['stdDevEvalValid' + thisModel] = stdDevEvalValid[thisModel]
    savemat(os.path.join(saveDirFigs, 'figVars.mat'), varsMatlab)

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

    #\\\ ACCURACY (Training and validation) for EACH MODEL
    for key in meanEvalTrain.keys():
        accFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanEvalTrain[key], yerr = stdDevEvalTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Accuracy')
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

    # ACCURACY (validation) for ALL MODELS
    allEvalValid = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanEvalValid.keys():
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Accuracy')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanEvalValid.keys()))
    allEvalValid.savefig(os.path.join(saveDirFigs,'allEvalValid.pdf'),
                    bbox_inches = 'tight')
