# 2018/10/02~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
model.py Model Module

Utilities useful for working on the model

Model: binds together the architecture, the loss function and the optimizer
"""

import os
import torch
import numpy as np
import pickle
import datetime

class Model:
    """
    Binds together in one class the architecture, the loss function and the
    optimizer. Printing an instance of the Model class gives all the information
    about the model.

    Attributes:

    archit: torch.nn.Module used for the architecture
    loss: torch loss function
    optim: torch optimizer
    name: model name
    saveDir: directory to save the model into
    nParameters: number of learnable parameters
        >> Obs.: the nParameters count is not accurate if filters are 
            Edge-Variant

    Methods:

    save(saveDir, label = ''[, saveDir = pathToDir]):
        Saves the architecture and optimization states in the directory
        specified by saveDir/savedModels. (Obs.: Directory 'savedModels' is
        created).
        The naming convention is name + 'Archit' + label + '.ckpt' for the
        architecture, and name + 'Optim' + label + '.ckpt' for the optimizer.
        In both cases, name is the name of the model used for initialization.
        Optionally, another saveDir can be specified (this saveDir does not
        override the saveDir stored when the model was created)

    load(label = '' [, loadFiles = (af, of)]):
        Loads the state of a saved architecture.
        If no loadFiles are specified, then the architecture is load from the
        directory previously specified by saveDir when .save() was called. If
        this is the case, the fetched files have to be in savedModels and have
        the name convention as specified in the .save() documentation.
        If loadFiles is provided, then the states provided in af file path are
        loaded for the architecture and in of file path for the optimizer. If
        loadFiles are specified, the input label is ignored.

    train(data, nEpochs, batchSize, [optionalArguments]):
        Trains the model.
        Input:
            data (class): contains the data, requires methods getSamples() and
                evaluate()
            nEpochs (int): number of epochs (passes through the dataset)
            batchSize (int): size of the batch
            [optionalArguments:]
            doLogging (bool): log the training run in tensorboard
                (default: False)
            doSaveVars (bool): save training variables (default: True)
            printInterval (int): how many training steps between priting the
                training state through the output (0 means not show anything)
                (default: (numberTrainingSamples//batchSize)//5)
            learningRateDecayRate (float): multiplier of the learning rate after
                each epoch
            learningRateDecayPeriod (int): after how many epochs update the
                learning rate
            >> Obs.: Both need to be specified for learning rate decay to take
                place, by default, there is no learning rate decay.
            validationInterval (int): every how many training steps to carry out
                a validation step (default: numberTrainingSamples//batchSize)
            earlyStoppingLag (int): how many steps after a best in validation
                has been found to stop the training (default: no early stopping)

    evaluate (data):
        After the model has been trained, evaluates the data, both on the best
        model (following validation) and on the last model.
        Input:
            data (class): contains the data, requires methods getSamples() and
                evaluate()
        Output:
            evalBest (scalar): Evaluation performance, following data.evaluate()
                for the best model
            evalLast (scalar): Evaluation performance, following data.evaluate()
                for the last model
                
    getTrainingOptions():
        Return the actual training options used for training. If no training
        has been done through the .train method, then returns None.
                
    Example (Single model training):
        (For multiple model training, refer to Modules.train)
        
    Once we have initialized the architecture (archit), the loss function (loss)
    and the optimizer (optim), and have determined a name, a save directory 
    (saveDir) and a node ordering (order), we initialize the model.
    
    thisModel = model.Model(archit, loss, optim, name, saveDir, order)
    
    Then, given the data (class with an .evaluate() and .getSamples() method as
    those defined in Utils.dataTools), the number of epochs (nEpochs), and the
    batch size (batchSize), together with the specific training options, we can
    train the model as
    
    thisModel.train(data, nEpochs, batchSize, [optional keyword arguments])
    
    Once the model is train, we can run the evaluation on the testing set
    (again, input the data class)
    
    evalBest, evalLast = thisModel.evaluate(data)
    
    Which prints the evaluation result and stores it in the output variables.
    """
    def __init__(self, architecture, loss, optimizer, name, saveDir):
        self.archit = architecture
        self.nParameters = 0
        # Count parameters:
        for param in list(self.archit.parameters()):
            if len(param.shape)>0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.nParameters += thisNParam
            else:
                pass
        self.loss = loss
        self.optim = optimizer
        self.name = name
        self.saveDir = saveDir
        self.trainingOptions = None

    def save(self, label = '', **kwargs):
        if 'saveDir' in kwargs.keys():
            saveDir = kwargs['saveDir']
        else:
            saveDir = self.saveDir
        saveModelDir = os.path.join(saveDir,'savedModels')
        # Create directory savedModels if it doesn't exist yet:
        if not os.path.exists(saveModelDir):
            os.makedirs(saveModelDir)
        saveFile = os.path.join(saveModelDir, self.name)
        torch.save(self.archit.state_dict(), saveFile+'Archit'+ label+'.ckpt')
        torch.save(self.optim.state_dict(), saveFile+'Optim'+label+'.ckpt')

    def load(self, label = '', **kwargs):
        if 'loadFiles' in kwargs.keys():
            (architLoadFile, optimLoadFile) = kwargs['loadFiles']
        else:
            saveModelDir = os.path.join(self.saveDir,'savedModels')
            architLoadFile = os.path.join(saveModelDir,
                                          self.name + 'Archit' + label +'.ckpt')
            optimLoadFile = os.path.join(saveModelDir,
                                         self.name + 'Optim' + label + '.ckpt')
        self.archit.load_state_dict(torch.load(architLoadFile))
        self.optim.load_state_dict(torch.load(optimLoadFile))
        
    def getTrainingOptions(self):
        
        return self.trainingOptions

    def train(self, data, nEpochs, batchSize, **kwargs):

        ####################################
        # ARGUMENTS (Store chosen options) #
        ####################################

        # Training Options:
        if 'doLogging' in kwargs.keys():
            doLogging = kwargs['doLogging']
        else:
            doLogging = False

        if 'doSaveVars' in kwargs.keys():
            doSaveVars = kwargs['doSaveVars']
        else:
            doSaveVars = True

        if 'printInterval' in kwargs.keys():
            printInterval = kwargs['printInterval']
            if printInterval > 0:
                doPrint = True
            else:
                doPrint = False
        else:
            doPrint = True
            printInterval = (data.nTrain//batchSize)//5

        if 'learningRateDecayRate' in kwargs.keys() and \
            'learningRateDecayPeriod' in kwargs.keys():
            doLearningRateDecay = True
            learningRateDecayRate = kwargs['learningRateDecayRate']
            learningRateDecayPeriod = kwargs['learningRateDecayPeriod']
        else:
            doLearningRateDecay = False

        if 'validationInterval' in kwargs.keys():
            validationInterval = kwargs['validationInterval']
        else:
            validationInterval = data.nTrain//batchSize

        if 'earlyStoppingLag' in kwargs.keys():
            doEarlyStopping = True
            earlyStoppingLag = kwargs['earlyStoppingLag']
        else:
            doEarlyStopping = False
            earlyStoppingLag = 0

        # No training case:
        if nEpochs == 0:
            doSaveVars = False
            doLogging = False
            # If there's no training happening, there's nothing to report about
            # training losses and stuff.

        if doLogging:
            from Utils.visualTools import Visualizer
            logsTB = os.path.join(self.saveDir, self.name + '-logsTB')
            logger = Visualizer(logsTB, name='visualResults')
        else:
            logger = None

        # Get the device we're working on
        device = None # Not set
        params = list(self.archit.parameters())
        thisDevice = params[0].device
        if device is None:
            device = thisDevice
        else:
            assert device == thisDevice

        ###########################################
        # DATA INPUT (pick up on data parameters) #
        ###########################################

        nTrain = data.nTrain # size of the training set

        # Number of batches: If the desired number of batches does not split the
        # dataset evenly, we reduce the size of the last batch (the number of
        # samples in the last batch).
        # The variable batchSize is a list of length nBatches (number of 
        # batches), where each element of the list is a number indicating the
        # size of the corresponding batch.
        if nTrain < batchSize:
            nBatches = 1
            batchSize = [nTrain]
        elif nTrain % batchSize != 0:
            nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
            batchSize = [batchSize] * nBatches
            # If the sum of all batches so far is not the total number of 
            # graphs, start taking away samples from the last batch (remember 
            # that we used ceiling, so we are overshooting with the estimated 
            # number of batches)
            while sum(batchSize) != nTrain:
                batchSize[-1] -= 1
        # If they fit evenly, then just do so.
        else:
            nBatches = np.int(nTrain/batchSize)
            batchSize = [batchSize] * nBatches
        # batchIndex is used to determine the first and last element of each
        # batch.
        # If batchSize is, for example [20,20,20] meaning that there are three
        # batches of size 20 each, then cumsum will give [20,40,60] which 
        # determines the last index of each batch: up to 20, from 20 to 40, and
        # from 40 to 60. We add the 0 at the beginning so that 
        # batchIndex[b]:batchIndex[b+1] gives the right samples for batch b.
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex

        ###################
        # SAVE ATTRIBUTES #
        ###################

        self.trainingOptions = {}
        self.trainingOptions['doLogging'] = doLogging
        self.trainingOptions['logger'] = logger
        self.trainingOptions['doSaveVars'] = doSaveVars
        self.trainingOptions['doPrint'] = printInterval
        self.trainingOptions['printInterval'] = printInterval
        self.trainingOptions['doLearningRateDecay'] = doLearningRateDecay
        if doLearningRateDecay:
            self.trainingOptions['learningRateDecayRate'] = \
                                                         learningRateDecayRate
            self.trainingOptions['learningRateDecayPeriod'] = \
                                                         learningRateDecayPeriod
        self.trainingOptions['validationInterval'] = validationInterval
        self.trainingOptions['doEarlyStopping'] = doEarlyStopping
        self.trainingOptions['earlyStoppingLag'] = earlyStoppingLag


        ##############
        # TRAINING   #
        ##############

        # Learning rate scheduler:
        if doLearningRateDecay:
            learningRateScheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                    learningRateDecayPeriod, learningRateDecayRate)

        # Initialize counters (since we give the possibility of early stopping,
        # we had to drop the 'for' and use a 'while' instead):
        epoch = 0 # epoch counter
        lagCount = 0 # lag counter for early stopping

        if doSaveVars:
            lossTrain = []
            evalTrain = []
            lossValid = []
            evalValid = []
            timeTrain = []
            timeValid = []

        while epoch < nEpochs \
                    and (lagCount < earlyStoppingLag or (not doEarlyStopping)):
            # The condition will be zero (stop), whenever one of the items of
            # the 'and' is zero. Therefore, we want this to stop only for epoch
            # counting when we are NOT doing early stopping. This can be 
            # achieved if the second element of the 'and' is always 1 (so that
            # the first element, the epoch counting, decides). In order to 
            # force the second element to be one whenever there is not early
            # stopping, we have an or, and force it to one. So, when we are not
            # doing early stopping, the variable 'not doEarlyStopping' is 1, 
            # and the result of the 'or' is 1 regardless of the lagCount. When
            # we do early stopping, then the variable 'not doEarlyStopping' is 
            # 0, and the value 1 for the 'or' gate is determined by the lag
            # count.
            # ALTERNATIVELY, we could just keep 'and lagCount<earlyStoppingLag'
            # and be sure that lagCount can only be increased whenever
            # doEarlyStopping is True. But I somehow figured out that would be
            # harder to maintain (more parts of the code to check if we are
            # accidentally increasing lagCount).

            # Randomize dataset for each epoch
            randomPermutation = np.random.permutation(nTrain)
            # Convert a numpy.array of numpy.int into a list of actual int.
            idxEpoch = [int(i) for i in randomPermutation]

            # Learning decay
            if doLearningRateDecay:
                learningRateScheduler.step()

                if doPrint:
                    # All the optimization have the same learning rate, so just
                    # print one of them
                    # TODO: Actually, they might be different, so I will need to
                    # print all of them.
                    print("Epoch %d, learning rate = %.8f" % (epoch+1,
                          learningRateScheduler.optim.param_groups[0]['lr']))

            # Initialize counter
            batch = 0 # batch counter
            while batch < nBatches \
                        and (lagCount<earlyStoppingLag or (not doEarlyStopping)):

                # Extract the adequate batch
                thisBatchIndices = idxEpoch[batchIndex[batch]
                                            : batchIndex[batch+1]]
                # Get the samples
                xTrain, yTrain = data.getSamples('train', thisBatchIndices)
                xTrain = xTrain.to(device)
                yTrain = yTrain.to(device)
                
                # Start measuring time
                startTime = datetime.datetime.now()

                # Reset gradients
                self.archit.zero_grad()

                # Obtain the output of the GNN
                yHatTrain = self.archit(xTrain)

                # Compute loss
                lossValueTrain = self.loss(yHatTrain, yTrain)

                # Compute gradients
                lossValueTrain.backward()

                # Optimize
                self.optim.step()
                
                # Finish measuring time
                endTime = datetime.datetime.now()
                
                timeElapsed = abs(endTime - startTime).total_seconds()

                # Compute the accuracy
                #   Note: Using yHatTrain.data creates a new tensor with the
                #   same value, but detaches it from the gradient, so that no
                #   gradient operation is taken into account here.
                #   (Alternatively, we could use a with torch.no_grad():)
                accTrain = data.evaluate(yHatTrain.data, yTrain)

                # Logging values
                if doLogging:
                    lossTrainTB = lossValueTrain.item()
                    evalTrainTB = accTrain.item()
                # Save values
                if doSaveVars:
                    lossTrain += [lossValueTrain.item()]
                    evalTrain += [accTrain.item()]
                    timeTrain += [timeElapsed]

                # Print:
                if doPrint:
                    if (epoch * nBatches + batch) % printInterval == 0:
                        print("(E: %2d, B: %3d) %6.4f / %7.4f - %6.4fs" % (
                                epoch+1, batch+1, accTrain,
                                lossValueTrain.item(), timeElapsed))
                
                # Delete variables to free space in CUDA memory
                del xTrain
                del yTrain
                del lossValueTrain
                del accTrain

                #\\\\\\\
                #\\\ TB LOGGING (for each batch)
                #\\\\\\\

                if doLogging:
                    logger.scalar_summary(mode = 'Training',
                                          epoch = epoch * nBatches + batch,
                                          **{'lossTrain': lossTrainTB,
                                           'evalTrain': evalTrainTB})

                #\\\\\\\
                #\\\ VALIDATION
                #\\\\\\\

                if (epoch * nBatches + batch) % validationInterval == 0:
                    # Validation:
                    xValid, yValid = data.getSamples('valid')
                    xValid = xValid.to(device)
                    yValid = yValid.to(device)
                    
                    # Start measuring time
                    startTime = datetime.datetime.now()
                    
                    # Under torch.no_grad() so that the computations carried out
                    # to obtain the validation accuracy are not taken into
                    # account to update the learnable parameters.
                    with torch.no_grad():
                        # Obtain the output of the GNN
                        yHatValid = self.archit(xValid)

                        # Compute loss
                        lossValueValid = self.loss(yHatValid, yValid)
                        
                        # Finish measuring time
                        endTime = datetime.datetime.now()
                        
                        timeElapsed = abs(endTime - startTime).total_seconds()

                        # Compute accuracy:
                        accValid = data.evaluate(yHatValid, yValid)

                        # Logging values
                        if doLogging:
                            lossValidTB = lossValueValid.item()
                            evalValidTB = accValid.item()
                        # Save values
                        if doSaveVars:
                            lossValid += [lossValueValid.item()]
                            evalValid += [accValid.item()]
                            timeValid += [timeElapsed]

                    # Print:
                    if doPrint:
                        print("[VALIDATION] %6.4f / %7.4f - %6.4fs" % (
                                accValid.item(), lossValueValid.item(),
                                timeElapsed))
                        
                    # Delete variables to free space in CUDA memory
                    del xValid
                    del yValid
                    del lossValueValid

                    if doLogging:
                        logger.scalar_summary(mode = 'Validation',
                                          epoch = epoch * nBatches + batch,
                                          **{'lossValid': lossValidTB,
                                           'evalValid': evalValidTB})

                    # No previous best option, so let's record the first trial
                    # as the best option
                    if epoch == 0 and batch == 0:
                        bestScore = accValid
                        bestEpoch, bestBatch = epoch, batch
                        # Save this model as the best (so far)
                        self.save(label = 'Best')
                        # Start the counter
                        if doEarlyStopping:
                            initialBest = True
                    else:
                        thisValidScore = accValid
                        if thisValidScore > bestScore:
                            bestScore = thisValidScore
                            bestEpoch, bestBatch = epoch, batch
                            if doPrint:
                                print("\t=> New best achieved: %.4f" % \
                                          (bestScore))
                            self.save(label = 'Best')
                            # Now that we have found a best that is not the
                            # initial one, we can start counting the lag (if
                            # needed)
                            initialBest = False
                            # If we achieved a new best, then we need to reset
                            # the lag count.
                            if doEarlyStopping:
                                lagCount = 0
                        # If we didn't achieve a new best, increase the lag
                        # count.
                        # Unless it was the initial best, in which case we 
                        # haven't found any best yet, so we shouldn't be doing
                        # the early stopping count.
                        elif doEarlyStopping and not initialBest:
                            lagCount += 1

                #\\\\\\\
                #\\\ END OF BATCH:
                #\\\\\\\

                #\\\ Increase batch count:
                batch += 1

            #\\\\\\\
            #\\\ END OF EPOCH:
            #\\\\\\\

            #\\\ Increase epoch count:
            epoch += 1
            
        #\\\ Save models:
        self.save(label = 'Last')

        #################
        # TRAINING OVER #
        #################

        if doSaveVars:
            # We convert the lists into np.arrays
            self.lossTrain = np.array(lossTrain)
            self.evalTrain = np.array(evalTrain)
            self.lossValid = np.array(lossValid)
            self.evalValid = np.array(evalValid)
            # And we would like to save all the relevant information from
            # training
            saveDirVars = os.path.join(self.saveDir, self.name + '-trainVars')
            if not os.path.exists(saveDirVars):
                os.makedirs(saveDirVars)
            pathToFile = os.path.join(saveDirVars,'trainVars.pkl')
            with open(pathToFile, 'wb') as trainVarsFile:
                pickle.dump(
                    {'nEpochs': nEpochs,
                     'nBatches': nBatches,
                     'batchSize': np.array(batchSize),
                     'batchIndex': np.array(batchIndex),
                     'lossTrain': lossTrain,
                     'evalTrain': evalTrain,
                     'lossValid': lossValid,
                     'evalValid': evalValid
                     }, trainVarsFile)

        # Now, if we didn't do any training (i.e. nEpochs = 0), then the last is
        # also the best.
        if nEpochs == 0:
            self.save(label = 'Best')
            self.save(label = 'Last')
            if doPrint:
                print("WARNING: No training. Best and Last models are the same.")

        # After training is done, reload best model before proceeding to 
        # evaluation:
        self.load(label = 'Best')
        self.bestBatch = bestBatch
        self.bestEpoch = bestEpoch

        #\\\ Print out best:
        if doPrint and nEpochs > 0:
            print("=> Best validation achieved (E: %d, B: %d): %.4f" % (
                    bestEpoch + 1, bestBatch + 1, bestScore))

    def evaluate(self, data):
    
        # Get the device we're working on
        device = None # Not set
        params = list(self.archit.parameters())
        thisDevice = params[0].device
        if device is None:
            device = thisDevice
        else:
            assert device == thisDevice
    
        ########
        # DATA #
        ########

        xTest, yTest = data.getSamples('test')
        xTest = xTest.to(device)
        yTest = yTest.to(device)

        ##############
        # BEST MODEL #
        ##############

        self.load(label = 'Best')

        if self.trainingOptions['doPrint']:
            print("Total testing accuracy (Best):", flush = True)

        with torch.no_grad():
            # Process the samples
            yHatTest = self.archit(xTest)
            # yHatTest is of shape
            #   testSize x numberOfClasses
            # We compute the accuracy
            accBest = data.evaluate(yHatTest, yTest)
            
        del yHatTest

        if self.trainingOptions['doPrint']:
            print("Evaluation (Best): %4.2f%%" % accBest)

        ##############
        # LAST MODEL #
        ##############

        self.load(label = 'Last')

        with torch.no_grad():
            # Process the samples
            yHatTest = self.archit(xTest)
            # yHatTest is of shape
            #   testSize x numberOfClasses
            # We compute the accuracy
            accLast = data.evaluate(yHatTest, yTest)

        if self.trainingOptions['doPrint']:
            print("Evaluation (Last): %4.2f%%" % accLast)
            
        del xTest, yTest, yHatTest
            
        return accBest, accLast


    def __repr__(self):
        reprString  = "Name: %s\n" % (self.name)
        reprString += "Number of learnable parameters: %d\n"%(self.nParameters)
        reprString += "\n"
        reprString += "Model architecture:\n"
        reprString += "----- -------------\n"
        reprString += "\n"
        reprString += repr(self.archit) + "\n"
        reprString += "\n"
        reprString += "Loss function:\n"
        reprString += "---- ---------\n"
        reprString += "\n"
        reprString += repr(self.loss) + "\n"
        reprString += "\n"
        reprString += "Optimizer:\n"
        reprString += "----------\n"
        reprString += "\n"
        reprString += repr(self.optim) + "\n"
        return reprString
