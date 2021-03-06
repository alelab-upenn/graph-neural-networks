# 2021/03/04~
# Luana Ruiz, rubruiz@seas.upenn.edu.
# Fernando Gama, fgama@seas.upenn.edu.

# Simulate the epidemic tracking problem. In this experiment, we compare GRNNs
# and gated GRNNs in a binary node classification problem modeling the spread of
# an epidemic on a high school friendship network. The epidemic data is generated
# by using the SIR model to simulate the spread of an infectious disease on the 
# friendship network. The disease is first recorded on day t=0, when each individual 
# node is infected with probability p_{seed}=0.05. On the days that follow, an 
# infected student can then spread the disease to their susceptible friends with 
# probability p_inf=0.3 each day. Infected students become immune after 4 days, 
# at which point they can no longer spread or contract the disease. 
# Given the state of each node at some point in time (susceptible, infected or 
# recovered), the binary node classification problem is to predict whether each 
# node in the network will have the disease (i.e., be infected) seqLen=8 days ahead.

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

thisFilename = 'epidemicGRNN' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + today
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
nValid = 120 # Number of validation samples
nTest = 200 # Number of testing samples
seqLen = 8 # Sequence length
seedProb = 0.05
infectionProb = 0.3
recoveryTime = 4

nDataRealizations = 10 # Number of data realizations

#\\\ Save values:
writeVarValues(varsFile, {'nTrain': nTrain,
                          'nValid': nValid,
                          'nTest': nTest,
                          'seqLen': seqLen,
                          'seedProb': seedProb,
                          'infectionProb': infectionProb,
                          'recoveryTime': recoveryTime,
                          'nDataRealizations':nDataRealizations,
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
lossFunction = loss.F1Score 

#\\\ Overall training options
nEpochs = 10 # Number of epochs
batchSize = 100 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'optimAlg': optimAlg,
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
doGRNN = True
doTimeGatedGRNN = True
doNodeGatedGRNN = True
doEdgeGatedGRNN = True

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

#\\\\\\\\\\\\
#\\\ MODEL 1: GRNN
#\\\\\\\\\\\\
    
#\\\ Basic parameters for all the Selection GNN architectures

modelGRNN = {}
modelGRNN['name'] = 'GRNN' # To be modified later on depending on the
    # specific ordering selected
modelGRNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                 else 'cpu'
                                 
#\\\ ARCHITECTURE
    
# Select architectural nn.Module to use
modelGRNN['archit'] = archit.GraphRecurrentNN
# Graph convolutional layers
modelGRNN['dimInputSignals'] = 1 # Number of features of x
modelGRNN['dimOutputSignals'] = 2 # Number of features of y
modelGRNN['dimHiddenSignals'] = 12 # Number of features of z
modelGRNN['nFilterTaps'] = [5,5] # Number of filter taps
modelGRNN['bias'] = True # Include bias
# Nonlinearity
modelGRNN['nonlinearityHidden'] = nn.Tanh()
modelGRNN['nonlinearityOutput'] = nn.ReLU()
modelGRNN['nonlinearityReadout'] = nn.ReLU()
# Readout layer
modelGRNN['dimReadout'] = []
# Graph Structure
modelGRNN['GSO'] = None # To be determined later on, based on data

#\\\ TRAINER

modelGRNN['trainer'] = training.Trainer

#\\\ EVALUATOR

modelGRNN['evaluator'] = evaluation.evaluate

if doGRNN:
    
    #\\\ Save Values:
    writeVarValues(varsFile, modelGRNN)
    modelList += [modelGRNN['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 2: Time-gated GRNN
#\\\\\\\\\\\\

if doTimeGatedGRNN:  
    
    modelTimeGatedGRNN = deepcopy(modelGRNN)

    modelTimeGatedGRNN['name'] = 'TimeGatedGRNN'
    modelTimeGatedGRNN['archit'] = archit.GatedGraphRecurrentNN
    modelTimeGatedGRNN['gateType'] = 'time'
    
    #\\\ Save Values:
    writeVarValues(varsFile, modelTimeGatedGRNN)
    modelList += [modelTimeGatedGRNN['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 3: Node-gated GRNN
#\\\\\\\\\\\\

if doNodeGatedGRNN:  
    
    modelNodeGatedGRNN = deepcopy(modelGRNN)

    modelNodeGatedGRNN['name'] = 'NodeGatedGRNN'
    modelNodeGatedGRNN['archit'] = archit.GatedGraphRecurrentNN
    modelNodeGatedGRNN['gateType'] = 'node'
    
    #\\\ Save Values:
    writeVarValues(varsFile, modelNodeGatedGRNN)
    modelList += [modelNodeGatedGRNN['name']]

#\\\\\\\\\\\\
#\\\ MODEL 4: Edge-gated GRNN
#\\\\\\\\\\\\

if doEdgeGatedGRNN:  
    
    modelEdgeGatedGRNN = deepcopy(modelGRNN)

    modelEdgeGatedGRNN['name'] = 'EdgeGatedGRNN'
    modelEdgeGatedGRNN['archit'] = archit.GatedGraphRecurrentNN
    modelEdgeGatedGRNN['gateType'] = 'edge'
    
    #\\\ Save Values:
    writeVarValues(varsFile, modelEdgeGatedGRNN)
    modelList += [modelEdgeGatedGRNN['name']]
    
###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 10 # After how many training steps, print the partial results
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
    torch.cuda.empty_cache()

#\\\ Notify of processing units
if doPrint:
    print("Selected devices:")
    for thisModel in modelList:
        modelDict = eval('model' + thisModel)
        print("\t%s: %s" % (thisModel, modelDict['device']))

#\\\ Logging options
if doLogging:
    from alegnn.utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')
    
#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
costBest = {} # Cost for the best model (Evaluation cost: Error rate)
costLast = {} # Cost for the last model
for thisModel in modelList: # Create an element for each split realization,
    costBest[thisModel] = [] 
    costLast[thisModel] = []

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
        lossTrain[thisModel] = [] 
        costTrain[thisModel] = [] 
        lossValid[thisModel] = [] 
        costValid[thisModel] = [] 


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

# And in case each model has specific training options, then we create a 
# separate dictionary per model.

trainingOptsPerModel= {}

#%%##################################################################
#                                                                   #
#                    DATA HANDLING                                  #
#                                                                   #
#####################################################################

#########
# GRAPH #
#########

# Create graph
Adj = alegnn.utils.dataTools.Epidemics.createGraph()
nNodes = Adj.shape[0]
graphOptions = {}
graphOptions['adjacencyMatrix'] = Adj
G = graphTools.Graph('adjacency', nNodes, graphOptions)
G.computeGFT() # Compute the eigendecomposition of the stored GSO

for realization in range(nDataRealizations):

    ############
    # DATASETS #
    ############
    
    data = alegnn.utils.dataTools.Epidemics(seqLen, seedProb, infectionProb,
                                            recoveryTime, nTrain, nValid, 
                                            nTest)
    data.astype(torch.float64)
    #data.to(device)
    data.expandDims() # Data are just graph processes, but the architectures 
        # require that the input signals are of the form B x T x F x N, so we
        # need to expand the middle dimensions to convert them from B x T x N 
        # to B x T x 1 x N

    #%%##################################################################
    #                                                                   #
    #                    MODELS INITIALIZATION                          #
    #                                                                   #
    #####################################################################
    
    # This is the dictionary where we store the models (in a model.Model
    # class, that is then passed to training).
    modelsGRNN = {}

    # If a new model is to be created, it should be called for here.
    
    if doPrint:
        print("Model initialization...", flush = True)
        
    for thisModel in modelList:
        
        # Get the corresponding parameter dictionary
        modelDict = deepcopy(eval('model' + thisModel))
        # and training options
        trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)
        
        # Now, this dictionary has all the hyperparameters that we need to
        # pass to the architecture function, but it also has other keys
        # that belong to the more general model (like 'name' or 'device'),
        # so we need to extract them and save them in seperate variables
        # for future use.
        thisName = modelDict.pop('name')
        callArchit = modelDict.pop('archit')
        thisDevice = modelDict.pop('device')
        thisTrainer = modelDict.pop('trainer')
        thisEvaluator = modelDict.pop('evaluator')
        
        # If more than one graph or data realization is going to be 
        # carried out, we are going to store all of thos models
        # separately, so that any of them can be brought back and
        # studied in detail.
        if nDataRealizations > 1:
            thisName += 'R%02d' % realization
            
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

        #\\\ GSO
        # Normalize adjacency
        S = G.S.copy()/np.max(np.real(G.E))
            
        modelDict['GSO'] = S
        
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
        thisLossFunction = lossFunction

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
        modelsGRNN[thisName] = modelCreated

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

    for thisModel in modelsGRNN.keys():
        
        if doPrint:
            print("Training model %s..." % thisModel)
         
        # Remember that modelsGNN.keys() has the split numbering as well as 
        # the name, while modelList has only the name. So we need to map 
        # the specific model for this specific split with the actual model
        # name, since there are several variables that are indexed by the
        # model name (for instance, the training options, or the
        # dictionaries saving the loss values)
        for m in modelList:
            if m in thisModel:
                modelName = m
    
        # Identify the specific graph and data realizations at training time
        if nDataRealizations > 1:
            trainingOptions['realizationNo'] = realization
        
        # Train the model
        thisTrainVars = modelsGRNN[thisModel].train(data,
                                                   nEpochs,
                                                   batchSize,
                                                   **trainingOptsPerModel[modelName])

        if doFigs:
        # Find which model to save the results (when having multiple
        # realizations)
            lossTrain[modelName].append(thisTrainVars['lossTrain'])
            costTrain[modelName].append(thisTrainVars['costTrain'])
            lossValid[modelName].append(thisTrainVars['lossValid'])
            costValid[modelName].append(thisTrainVars['costValid'])
                    
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

    if doPrint:
        print("\nTotal testing error rate", end = '', flush = True)
        if nDataRealizations > 1:
            print(" (", end = '', flush = True)
            print("Realization %02d" % realization, end = '', flush = True)
            print(")", end = '', flush = True)
        print(":", flush = True)
        

    for thisModel in modelsGRNN.keys():
        
        # Same as before, separate the model name from the data or graph
        # realization number
        for m in modelList:
            if m in thisModel:
                modelName = m

        # Evaluate the model
        thisEvalVars = modelsGRNN[thisModel].evaluate(data)
        
        # Save the outputs
        thisCostBest = thisEvalVars['costBest']
        thisCostLast = thisEvalVars['costLast']
        
        # Write values
        writeVarValues(varsFile,
                       {'costBest%s' % thisModel: thisCostBest,
                        'costLast%s' % thisModel: thisCostLast})

        # Now check which is the model being trained
        costBest[modelName].append(thisCostBest)
        costLast[modelName].append(thisCostLast)
        # This is so that we can later compute a total accuracy with
        # the corresponding error.
        
        if doPrint:
            print("\t%s: %1.4f [Best] %1.4f [Last]" % (thisModel,
                                                           thisCostBest,
                                                           thisCostLast))

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanCostBestPerGraph = {} # Compute the mean accuracy (best) across all
    # realizations data realizations of a graph
meanCostLastPerGraph = {} # Compute the mean accuracy (last) across all
    # realizations data realizations of a graph
meanCostBest = {} # Mean across graphs (after having averaged across data
    # realizations)
meanCostLast = {} # Mean across graphs
stdDevCostBest = {} # Standard deviation across graphs
stdDevCostLast = {} # Standard deviation across graphs

if doPrint:
    print("\nFinal evaluations (%02d realizations)" % (nDataRealizations))

for thisModel in modelList:
    # Convert the lists nDataRealizations array
    costBest[thisModel] = np.array(costBest[thisModel])
    costLast[thisModel] = np.array(costLast[thisModel])
    
    if nDataRealizations == 1:
        meanCostBest[thisModel] = np.squeeze(costBest[thisModel])
        meanCostLast[thisModel] = np.squeeze(costLast[thisModel])
    else:  
        meanCostBest[thisModel] = np.mean(costBest[thisModel])
        meanCostLast[thisModel] = np.mean(costLast[thisModel])
        stdDevCostBest[thisModel] = np.std(costBest[thisModel])
        stdDevCostLast[thisModel] = np.std(costLast[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %1.4f (+-%1.4f) [Best] %1.4f (+-%1.4f) [Last]" % (
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
    
with open(varsFile, 'a+') as file:
    file.write("Final evaluations (%02d realizations)\n" % (nDataRealizations))
    for thisModel in modelList:
        file.write("\t%s: %1.4f (+-%1.4f) [Best] %1.4f (+-%1.4f) [Last]\n" % (
                   thisModel,
                   meanCostBest[thisModel],
                   stdDevCostBest[thisModel],
                   meanCostLast[thisModel],
                   stdDevCostLast[thisModel]))
    file.write('\n')

# FIX

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
        # And compute the statistics
        meanLossTrain[thisModel] = \
                            np.mean(np.array(lossTrain[thisModel]), axis = 0)
        meanCostTrain[thisModel] = \
                            np.mean(np.array(costTrain[thisModel]), axis = 0)
        meanLossValid[thisModel] = \
                            np.mean(np.array(lossValid[thisModel]), axis = 0)
        meanCostValid[thisModel] = \
                            np.mean(np.array(costValid[thisModel]), axis = 0)
        stdDevLossTrain[thisModel] = \
                            np.std(np.array(lossTrain[thisModel]), axis = 0)
        stdDevCostTrain[thisModel] = \
                            np.std(np.array(costTrain[thisModel]), axis = 0)
        stdDevLossValid[thisModel] = \
                            np.std(np.array(lossValid[thisModel]), axis = 0)
        stdDevCostValid[thisModel] = \
                            np.std(np.array(costValid[thisModel]), axis = 0)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
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
        plt.ylabel(r'Error rate')
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
    plt.ylabel(r'Error rate')
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
