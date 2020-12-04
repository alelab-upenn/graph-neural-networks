# 2020/02/25~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
training.py Training Module

Trainer classes

Trainer: general trainer that just computes a loss over a training set and
    runs an evaluation on a validation test
TrainerSingleNode: trainer class that computes a loss over the training set and
    runs an evaluation on a validation set, but assuming that the architectures
    involved have a single node forward structure and that the data involved
    has a method for identifying the target nodes
TrainerFlocking: traininer class that computes a loss over the training set,
    suited for the problem of flocking (i.e. it involves specific uses of
    the data, like computing trajectories or using DAGger)

"""

import torch
import numpy as np
import os
import pickle
import datetime

from alegnn.utils.dataTools import invertTensorEW

class Trainer:
    """
    Trainer: general trainer that just computes a loss over a training set and
        runs an evaluation on a validation test
        
    Initialization:
        
        model (Modules.model class): model to train
        data (Utils.data class): needs to have a getSamples and an evaluate
            method
        nEpochs (int): number of epochs (passes over the dataset)
        batchSize (int): size of each minibatch

        Optional (keyword) arguments:
            
        validationInterval (int): interval of training (number of training
            steps) without running a validation stage.

        learningRateDecayRate (float): float that multiplies the latest learning
            rate used.
        learningRateDecayPeriod (int): how many training steps before 
            multiplying the learning rate decay rate by the actual learning
            rate.
        > Obs.: Both of these have to be defined for the learningRateDecay
              scheduler to be activated.
        logger (Visualizer): save tensorboard logs.
        saveDir (string): path to the directory where to save relevant training
            variables.
        printInterval (int): how many training steps after which to print
            partial results (0 means do not print)
        graphNo (int): keep track of what graph realization this is
        realitizationNo (int): keep track of what data realization this is
        >> Alternatively, these last two keyword arguments can be used to keep
            track of different trainings of the same model
            
    Training:
        
        .train(): trains the model and returns trainVars dict with the keys
            'nEpochs': number of epochs (int)
            'nBatches': number of batches (int)
            'validationInterval': number of training steps in between 
                validation steps (int)
            'batchSize': batch size of each training step (np.array)
            'batchIndex': indices for the start sample and end sample of each
                batch (np.array)
            'lossTrain': loss function on the training samples for each training
                step (np.array)
            'evalTrain': evaluation function on the training samples for each
                training step (np.array)
            'lossValid': loss function on the validation samples for each
                validation step (np.array)
            'evalValid': evaluation function on the validation samples for each
                validation step (np.array)
    """
    
    def __init__(self, model, data, nEpochs, batchSize, **kwargs):
        
        #\\\ Store model
        
        self.model = model
        self.data = data
        
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

        if 'graphNo' in kwargs.keys():
            graphNo = kwargs['graphNo']
        else:
            graphNo = -1

        if 'realizationNo' in kwargs.keys():
            if 'graphNo' in kwargs.keys():
                realizationNo = kwargs['realizationNo']
            else:
                graphNo = kwargs['realizationNo']
                realizationNo = -1
        else:
            realizationNo = -1

        if doLogging:
            from alegnn.utils.visualTools import Visualizer
            logsTB = os.path.join(self.saveDir, self.name + '-logsTB')
            logger = Visualizer(logsTB, name='visualResults')
        else:
            logger = None
        
        # No training case:
        if nEpochs == 0:
            doSaveVars = False
            doLogging = False
            # If there's no training happening, there's nothing to report about
            # training losses and stuff.
            
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
        self.trainingOptions['doPrint'] = doPrint
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
        self.trainingOptions['batchIndex'] = batchIndex
        self.trainingOptions['batchSize'] = batchSize
        self.trainingOptions['nEpochs'] = nEpochs
        self.trainingOptions['nBatches'] = nBatches
        self.trainingOptions['graphNo'] = graphNo
        self.trainingOptions['realizationNo'] = realizationNo
        
    def trainBatch(self, thisBatchIndices):
        
        # Get the samples
        xTrain, yTrain = self.data.getSamples('train', thisBatchIndices)
        xTrain = xTrain.to(self.model.device)
        yTrain = yTrain.to(self.model.device)

        # Start measuring time
        startTime = datetime.datetime.now()

        # Reset gradients
        self.model.archit.zero_grad()

        # Obtain the output of the GNN
        yHatTrain = self.model.archit(xTrain)

        # Compute loss
        lossValueTrain = self.model.loss(yHatTrain, yTrain)

        # Compute gradients
        lossValueTrain.backward()

        # Optimize
        self.model.optim.step()

        # Finish measuring time
        endTime = datetime.datetime.now()

        timeElapsed = abs(endTime - startTime).total_seconds()

        # Compute the accuracy
        #   Note: Using yHatTrain.data creates a new tensor with the
        #   same value, but detaches it from the gradient, so that no
        #   gradient operation is taken into account here.
        #   (Alternatively, we could use a with torch.no_grad():)
        costTrain = self.data.evaluate(yHatTrain.data, yTrain)
        
        return lossValueTrain.item(), costTrain.item(), timeElapsed
    
    def validationStep(self):
        
        # Validation:
        xValid, yValid = self.data.getSamples('valid')
        xValid = xValid.to(self.model.device)
        yValid = yValid.to(self.model.device)

        # Start measuring time
        startTime = datetime.datetime.now()

        # Under torch.no_grad() so that the computations carried out
        # to obtain the validation accuracy are not taken into
        # account to update the learnable parameters.
        with torch.no_grad():
            # Obtain the output of the GNN
            yHatValid = self.model.archit(xValid)

            # Compute loss
            lossValueValid = self.model.loss(yHatValid, yValid)

            # Finish measuring time
            endTime = datetime.datetime.now()

            timeElapsed = abs(endTime - startTime).total_seconds()

            # Compute accuracy:
            costValid = self.data.evaluate(yHatValid, yValid)
        
        return lossValueValid.item(), costValid.item(), timeElapsed
        
    def train(self):
        
        # Get back the training options
        assert 'trainingOptions' in dir(self)
        assert 'doLogging' in self.trainingOptions.keys()
        doLogging = self.trainingOptions['doLogging']
        assert 'logger' in self.trainingOptions.keys()
        logger = self.trainingOptions['logger']
        assert 'doSaveVars' in self.trainingOptions.keys()
        doSaveVars = self.trainingOptions['doSaveVars']
        assert 'doPrint' in self.trainingOptions.keys()
        doPrint = self.trainingOptions['doPrint']
        assert 'printInterval' in self.trainingOptions.keys()
        printInterval = self.trainingOptions['printInterval']
        assert 'doLearningRateDecay' in self.trainingOptions.keys()
        doLearningRateDecay = self.trainingOptions['doLearningRateDecay']
        if doLearningRateDecay:
            assert 'learningRateDecayRate' in self.trainingOptions.keys()
            learningRateDecayRate=self.trainingOptions['learningRateDecayRate']
            assert 'learningRateDecayPeriod' in self.trainingOptions.keys()
            learningRateDecayPeriod=self.trainingOptions['learningRateDecayPeriod']
        assert 'validationInterval' in self.trainingOptions.keys()
        validationInterval = self.trainingOptions['validationInterval']
        assert 'doEarlyStopping' in self.trainingOptions.keys()
        doEarlyStopping = self.trainingOptions['doEarlyStopping']
        assert 'earlyStoppingLag' in self.trainingOptions.keys()
        earlyStoppingLag = self.trainingOptions['earlyStoppingLag']
        assert 'batchIndex' in self.trainingOptions.keys()
        batchIndex = self.trainingOptions['batchIndex']
        assert 'batchSize' in self.trainingOptions.keys()
        batchSize = self.trainingOptions['batchSize']
        assert 'nEpochs' in self.trainingOptions.keys()
        nEpochs = self.trainingOptions['nEpochs']
        assert 'nBatches' in self.trainingOptions.keys()
        nBatches = self.trainingOptions['nBatches']
        assert 'graphNo' in self.trainingOptions.keys()
        graphNo = self.trainingOptions['graphNo']
        assert 'realizationNo' in self.trainingOptions.keys()
        realizationNo = self.trainingOptions['realizationNo']
        
        # Learning rate scheduler:
        if doLearningRateDecay:
            learningRateScheduler = torch.optim.lr_scheduler.StepLR(
                 self.model.optim,learningRateDecayPeriod,learningRateDecayRate)

        # Initialize counters (since we give the possibility of early stopping,
        # we had to drop the 'for' and use a 'while' instead):
        epoch = 0 # epoch counter
        lagCount = 0 # lag counter for early stopping
        
        # Store the training variables
        lossTrain = []
        costTrain = []
        lossValid = []
        costValid = []
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
            randomPermutation = np.random.permutation(self.data.nTrain)
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
                
                lossValueTrain, costValueTrain, timeElapsed = \
                                               self.trainBatch(thisBatchIndices)
                

                # Logging values
                if doLogging:
                    lossTrainTB = lossValueTrain
                    costTrainTB = costValueTrain
                # Save values
                lossTrain += [lossValueTrain]
                costTrain += [costValueTrain]
                timeTrain += [timeElapsed]

                # Print:
                if doPrint:
                    if (epoch * nBatches + batch) % printInterval == 0:
                        print("\t(E: %2d, B: %3d) %6.4f / %7.4f - %6.4fs" % (
                                epoch+1, batch+1, costValueTrain,
                                lossValueTrain, timeElapsed),
                            end = ' ')
                        if graphNo > -1:
                            print("[%d" % graphNo, end = '')
                            if realizationNo > -1:
                                print("/%d" % realizationNo,
                                      end = '')
                            print("]", end = '')
                        print("")

                #\\\\\\\
                #\\\ TB LOGGING (for each batch)
                #\\\\\\\

                if doLogging:
                    logger.scalar_summary(mode = 'Training',
                                          epoch = epoch * nBatches + batch,
                                          **{'lossTrain': lossTrainTB,
                                           'costTrain': costTrainTB})

                #\\\\\\\
                #\\\ VALIDATION
                #\\\\\\\

                if (epoch * nBatches + batch) % validationInterval == 0:

                    lossValueValid, costValueValid, timeElapsed = \
                                                           self.validationStep()

                    # Logging values
                    if doLogging:
                        lossValidTB = lossValueValid
                        costValidTB = costValueValid
                    # Save values
                    lossValid += [lossValueValid]
                    costValid += [costValueValid]
                    timeValid += [timeElapsed]

                    # Print:
                    if doPrint:
                        print("\t(E: %2d, B: %3d) %6.4f / %7.4f - %6.4fs" % (
                                epoch+1, batch+1,
                                costValueValid, 
                                lossValueValid,
                                timeElapsed), end = ' ')
                        print("[VALIDATION", end = '')
                        if graphNo > -1:
                            print(".%d" % graphNo, end = '')
                            if realizationNo > -1:
                                print("/%d" % realizationNo, end = '')
                        print(" (%s)]" % self.model.name)


                    if doLogging:
                        logger.scalar_summary(mode = 'Validation',
                                          epoch = epoch * nBatches + batch,
                                          **{'lossValid': lossValidTB,
                                           'costValid': costValidTB})

                    # No previous best option, so let's record the first trial
                    # as the best option
                    if epoch == 0 and batch == 0:
                        bestScore = costValueValid
                        bestEpoch, bestBatch = epoch, batch
                        # Save this model as the best (so far)
                        self.model.save(label = 'Best')
                        # Start the counter
                        if doEarlyStopping:
                            initialBest = True
                    else:
                        thisValidScore = costValueValid
                        if thisValidScore < bestScore:
                            bestScore = thisValidScore
                            bestEpoch, bestBatch = epoch, batch
                            if doPrint:
                                print("\t=> New best achieved: %.4f" % \
                                          (bestScore))
                            self.model.save(label = 'Best')
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
        self.model.save(label = 'Last')

        #################
        # TRAINING OVER #
        #################

        # We convert the lists into np.arrays
        lossTrain = np.array(lossTrain)
        costTrain = np.array(costTrain)
        lossValid = np.array(lossValid)
        costValid = np.array(costValid)
        # And we would like to save all the relevant information from
        # training
        trainVars = {'nEpochs': nEpochs,
                     'nBatches': nBatches,
                     'validationInterval': validationInterval,
                     'batchSize': np.array(batchSize),
                     'batchIndex': np.array(batchIndex),
                     'lossTrain': lossTrain,
                     'costTrain': costTrain,
                     'lossValid': lossValid,
                     'costValid': costValid
                     }
        
        if doSaveVars:
            saveDirVars = os.path.join(self.model.saveDir, 'trainVars')
            if not os.path.exists(saveDirVars):
                os.makedirs(saveDirVars)
            pathToFile = os.path.join(saveDirVars,
                                      self.model.name + 'trainVars.pkl')
            with open(pathToFile, 'wb') as trainVarsFile:
                pickle.dump(trainVars, trainVarsFile)

        # Now, if we didn't do any training (i.e. nEpochs = 0), then the last is
        # also the best.
        if nEpochs == 0:
            self.model.save(label = 'Best')
            self.model.save(label = 'Last')
            if doPrint:
                print("WARNING: No training. Best and Last models are the same.")

        # After training is done, reload best model before proceeding to
        # evaluation:
        self.model.load(label = 'Best')

        #\\\ Print out best:
        if doPrint and nEpochs > 0:
            print("=> Best validation achieved (E: %d, B: %d): %.4f" % (
                    bestEpoch + 1, bestBatch + 1, bestScore))
            
        return trainVars
    
class TrainerSingleNode(Trainer):
    """
    TrainerSingleNode: trainer class that computes a loss over the training set
        and runs an evaluation on a validation set, but assuming that the
        architectures involved have a single node forward structure and that the
        data involved has a method for identifying the target nodes
        
    Initialization:
        
        model (Modules.model class): model to train
        data (Utils.data class): needs to have a getSamples and an evaluate
            method
        nEpochs (int): number of epochs (passes over the dataset)
        batchSize (int): size of each minibatch

        Optional (keyword) arguments:
            
        validationInterval (int): interval of training (number of training
            steps) without running a validation stage.

        learningRateDecayRate (float): float that multiplies the latest learning
            rate used.
        learningRateDecayPeriod (int): how many training steps before 
            multiplying the learning rate decay rate by the actual learning
            rate.
        > Obs.: Both of these have to be defined for the learningRateDecay
              scheduler to be activated.
        logger (Visualizer): save tensorboard logs.
        saveDir (string): path to the directory where to save relevant training
            variables.
        printInterval (int): how many training steps after which to print
            partial results (0 means do not print)
        graphNo (int): keep track of what graph realization this is
        realitizationNo (int): keep track of what data realization this is
        >> Alternatively, these last two keyword arguments can be used to keep
            track of different trainings of the same model
            
    Training:
        
        .train(): trains the model and returns trainVars dict with the keys
            'nEpochs': number of epochs (int)
            'nBatches': number of batches (int)
            'validationInterval': number of training steps in between 
                validation steps (int)
            'batchSize': batch size of each training step (np.array)
            'batchIndex': indices for the start sample and end sample of each
                batch (np.array)
            'lossTrain': loss function on the training samples for each training
                step (np.array)
            'evalTrain': evaluation function on the training samples for each
                training step (np.array)
            'lossValid': loss function on the validation samples for each
                validation step (np.array)
            'evalValid': evaluation function on the validation samples for each
                validation step (np.array)
    """
    
    def __init__(self, model, data, nEpochs, batchSize, **kwargs):
        
        assert 'singleNodeForward' in dir(model.archit)
        assert 'getLabelID' in dir(data)
        
        # Initialize supraclass
        super().__init__(model, data, nEpochs, batchSize, **kwargs)
        
    def trainBatch(self, thisBatchIndices):
        
        # Get the samples
        xTrain, yTrain = self.data.getSamples('train', thisBatchIndices)
        xTrain = xTrain.to(self.model.device)
        yTrain = yTrain.to(self.model.device)
        targetIDs = self.data.getLabelID('train', thisBatchIndices)

        # Start measuring time
        startTime = datetime.datetime.now()

        # Reset gradients
        self.model.archit.zero_grad()

        # Obtain the output of the GNN
        yHatTrain = self.model.archit.singleNodeForward(xTrain, targetIDs)

        # Compute loss
        lossValueTrain = self.model.loss(yHatTrain, yTrain)

        # Compute gradients
        lossValueTrain.backward()

        # Optimize
        self.model.optim.step()

        # Finish measuring time
        endTime = datetime.datetime.now()

        timeElapsed = abs(endTime - startTime).total_seconds()

        # Compute the accuracy
        #   Note: Using yHatTrain.data creates a new tensor with the
        #   same value, but detaches it from the gradient, so that no
        #   gradient operation is taken into account here.
        #   (Alternatively, we could use a with torch.no_grad():)
        costTrain = self.data.evaluate(yHatTrain.data, yTrain)
        
        return lossValueTrain.item(), costTrain.item(), timeElapsed
    
    def validationStep(self):
        
        # Validation:
        xValid, yValid = self.data.getSamples('valid')
        xValid = xValid.to(self.model.device)
        yValid = yValid.to(self.model.device)
        targetIDs = self.data.getLabelID('valid')

        # Start measuring time
        startTime = datetime.datetime.now()

        # Under torch.no_grad() so that the computations carried out
        # to obtain the validation accuracy are not taken into
        # account to update the learnable parameters.
        with torch.no_grad():
            # Obtain the output of the GNN
            yHatValid = self.model.archit.singleNodeForward(xValid, targetIDs)

            # Compute loss
            lossValueValid = self.model.loss(yHatValid, yValid)

            # Finish measuring time
            endTime = datetime.datetime.now()

            timeElapsed = abs(endTime - startTime).total_seconds()

            # Compute accuracy:
            costValid = self.data.evaluate(yHatValid, yValid)
        
        return lossValueValid.item(), costValid.item(), timeElapsed
        
class TrainerFlocking(Trainer):
    """
    Trainer: trains flocking models, following the appropriate evaluation of
        the cost, and has options for different DAGger alternatives
        
    Initialization:
        
        model (Modules.model class): model to train
        data (Utils.data class): needs to have a getSamples and an evaluate
            method
        nEpochs (int): number of epochs (passes over the dataset)
        batchSize (int): size of each minibatch

        Optional (keyword) arguments:
        
        probExpert (float): initial probability of choosing the expert
        DAGgerType ('fixedBatch', 'randomEpoch', 'replaceTimeBatch'):
            'fixedBatch' (default if 'probExpert' is defined): doubles the batch
                samples by considering the same initial velocities and 
                positions, a trajectory given by the latest trained
                architecture, and the corresponding correction given by the
                optimal acceleration (i.e. for each position and velocity we 
                give what would be the optimal acceleration, even though the
                next position and velocity won't reflect this decision, but the
                one taken by the learned policy)
            'randomEpoch':  forms a new training set for each epoch consisting,
                with probability probExpert, of samples of the original dataset
                (optimal trajectories) and with probability 1-probExpert, with
                trajectories following the latest trained dataset.
            'replaceTimeBatch': creates a fixed number of new trajectories
                following randomly at each time step either the optimal control
                or the learned control; then, replaces this fixed number of new
                trajectores into the training set (then these might, or might 
                not get selected by the next batch)
            
        validationInterval (int): interval of training (number of training
            steps) without running a validation stage.

        learningRateDecayRate (float): float that multiplies the latest learning
            rate used.
        learningRateDecayPeriod (int): how many training steps before 
            multiplying the learning rate decay rate by the actual learning
            rate.
        > Obs.: Both of these have to be defined for the learningRateDecay
              scheduler to be activated.
        logger (Visualizer): save tensorboard logs.
        saveDir (string): path to the directory where to save relevant training
            variables.
        printInterval (int): how many training steps after which to print
            partial results (0 means do not print)
        graphNo (int): keep track of what graph realization this is
        realitizationNo (int): keep track of what data realization this is
        >> Alternatively, these last two keyword arguments can be used to keep
            track of different trainings of the same model
            
    Training:
        
        .train(): trains the model and returns trainVars dict with the keys
            'nEpochs': number of epochs (int)
            'nBatches': number of batches (int)
            'validationInterval': number of training steps in between 
                validation steps (int)
            'batchSize': batch size of each training step (np.array)
            'batchIndex': indices for the start sample and end sample of each
                batch (np.array)
            'bestBatch': batch index at which the best model was achieved (int)
            'bestEpoch': epoch at which the best model was achieved (int)
            'bestScore': evaluation measure on the validation sample that 
                achieved the best model (i.e. minimum achieved evaluation
                measure on the validation set)
            'lossTrain': loss function on the training samples for each training
                step (np.array)
            'timeTrain': time elapsed at each training step (np.array)
            'evalValid': evaluation function on the validation samples for each
                validation step (np.array)
            'timeValid': time elapsed at each validation step (np.array)
    """
    
    def __init__(self, model, data, nEpochs, batchSize, **kwargs):
        
        # Initialize supraclass
        super().__init__(model, data, nEpochs, batchSize, **kwargs)
        
        # Add the specific options
        
        if 'probExpert' in kwargs.keys():
            doDAGger = True
            probExpert = kwargs['probExpert']
        else:
            doDAGger = False
        
        if 'DAGgerType' in kwargs.keys():
            DAGgerType = kwargs['DAGgerType']
        else:
            DAGgerType = 'fixedBatch'
                
        self.trainingOptions['doDAGger'] = doDAGger
        if doDAGger:
            self.trainingOptions['probExpert'] = probExpert
            self.trainingOptions['DAGgerType'] = DAGgerType

    def train(self):
        
        # Get back the training options
        assert 'trainingOptions' in dir(self)
        assert 'doLogging' in self.trainingOptions.keys()
        doLogging = self.trainingOptions['doLogging']
        assert 'logger' in self.trainingOptions.keys()
        logger = self.trainingOptions['logger']
        assert 'doSaveVars' in self.trainingOptions.keys()
        doSaveVars = self.trainingOptions['doSaveVars']
        assert 'doPrint' in self.trainingOptions.keys()
        doPrint = self.trainingOptions['doPrint']
        assert 'printInterval' in self.trainingOptions.keys()
        printInterval = self.trainingOptions['printInterval']
        assert 'doLearningRateDecay' in self.trainingOptions.keys()
        doLearningRateDecay = self.trainingOptions['doLearningRateDecay']
        if doLearningRateDecay:
            assert 'learningRateDecayRate' in self.trainingOptions.keys()
            learningRateDecayRate=self.trainingOptions['learningRateDecayRate']
            assert 'learningRateDecayPeriod' in self.trainingOptions.keys()
            learningRateDecayPeriod=self.trainingOptions['learningRateDecayPeriod']
        assert 'validationInterval' in self.trainingOptions.keys()
        validationInterval = self.trainingOptions['validationInterval']
        assert 'doEarlyStopping' in self.trainingOptions.keys()
        doEarlyStopping = self.trainingOptions['doEarlyStopping']
        assert 'earlyStoppingLag' in self.trainingOptions.keys()
        earlyStoppingLag = self.trainingOptions['earlyStoppingLag']
        assert 'batchIndex' in self.trainingOptions.keys()
        batchIndex = self.trainingOptions['batchIndex']
        assert 'batchSize' in self.trainingOptions.keys()
        batchSize = self.trainingOptions['batchSize']
        assert 'nEpochs' in self.trainingOptions.keys()
        nEpochs = self.trainingOptions['nEpochs']
        assert 'nBatches' in self.trainingOptions.keys()
        nBatches = self.trainingOptions['nBatches']
        assert 'graphNo' in self.trainingOptions.keys()
        graphNo = self.trainingOptions['graphNo']
        assert 'realizationNo' in self.trainingOptions.keys()
        realizationNo = self.trainingOptions['realizationNo']
        assert 'doDAGger' in self.trainingOptions.keys()
        doDAGger = self.trainingOptions['doDAGger']
        if doDAGger:
            assert 'DAGgerType' in self.trainingOptions.keys()
            DAGgerType = self.trainingOptions['DAGgerType']
        
        # Get the values we need
        nTrain = self.data.nTrain
        thisArchit = self.model.archit
        thisLoss = self.model.loss
        thisOptim = self.model.optim
        thisDevice = self.model.device
        
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
            evalValid = []
            timeTrain = []
            timeValid = []

        # Get original dataset
        xTrainOrig, yTrainOrig = self.data.getSamples('train')
        StrainOrig = self.data.getData('commGraph', 'train')
        initVelTrainAll = self.data.getData('initVel', 'train')
        if doDAGger:
            initPosTrainAll = self.data.getData('initPos', 'train')

        # And save it as the original "all samples"
        xTrainAll = xTrainOrig
        yTrainAll = yTrainOrig
        StrainAll = StrainOrig

        # If it is:
        #   'randomEpoch' assigns always the original training set at the
        #       beginning of each epoch, so it is reset by using the variable
        #       Orig, instead of the variable all
        #   'replaceTimeBatch' keeps working only in the All variables, so
        #       every epoch updates the previous dataset, and never goes back
        #       to the original dataset (i.e. there is no Orig involved in
        #       the 'replaceTimeBatch' DAGger)
        #   'fixedBatch': it takes All = Orig from the beginning and then it
        #       doesn't matter becuase it always acts by creating a new
        #       batch with "corrected" trajectories for the learned policies

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
                    
            #\\\\\\\\\\\\\\\\
            #\\\ Start DAGGER: randomEpoch
            #\\\
            if doDAGger and epoch > 0 and DAGgerType == 'randomEpoch':

                # The 'randomEpoch' option forms a new training set for each
                # epoch consisting, with probability probExpert, of samples
                # of the original dataset (optimal trajectories) and with
                # probability 1-probExpert, with trajectories following the
                # latest trained dataset.

                xTrainAll, yTrainAll, StrainAll = \
                    self.randomEpochDAGger(epoch, xTrainOrig, yTrainOrig,
                                           StrainOrig, initPosTrainAll,
                                           initVelTrainAll)
            #\\\
            #\\\ Finished DAGGER
            #\\\\\\\\\\\\\\\\\\\

            # Initialize counter
            batch = 0 # batch counter
            while batch < nBatches \
                      and (lagCount<earlyStoppingLag or (not doEarlyStopping)):
                          
                #\\\\\\\\\\\\\\\\
                #\\\ Start DAGGER: replaceTimeBatch
                #\\\
                if doDAGger and (batch > 0 or epoch > 0)\
                                          and DAGgerType == 'replaceTimeBatch':

                    # The option 'replaceTimeBatch' creates a fixed number of
                    # new trajectories following randomly at each time step
                    # either the optimal control or the learned control
                    # Then, replaces this fixed number of new trajectores into
                    # the training set (then these might, or might not get
                    # selected by the next batch)

                    xTrainAll, yTrainAll, StrainAll = \
                        self.replaceTimeBatchDAGger(epoch, xTrainAll, yTrainAll,
                                                    StrainAll, initPosTrainAll,
                                                    initVelTrainAll)
                #\\\
                #\\\ Finished DAGGER
                #\\\\\\\\\\\\\\\\\\\

                # Extract the adequate batch
                thisBatchIndices = idxEpoch[batchIndex[batch]
                                            : batchIndex[batch+1]]
                # Get the samples
                xTrain = xTrainAll[thisBatchIndices]
                yTrain = yTrainAll[thisBatchIndices]
                Strain = StrainAll[thisBatchIndices]
                initVelTrain = initVelTrainAll[thisBatchIndices]
                if doDAGger and DAGgerType == 'fixedBatch':
                    initPosTrain = initPosTrainAll[thisBatchIndices]
                    
                #\\\\\\\\\\\\\\\\
                #\\\ Start DAGGER: fixedBatch
                #\\\
                if doDAGger and (batch > 0 or epoch > 0)\
                                                and DAGgerType == 'fixedBatch':

                    # The 'fixedBatch' option, doubles the batch samples
                    # by considering the same initial velocities and
                    # positions, a trajectory given by the latest trained
                    # architecture, and the corresponding correction
                    # given by the optimal acceleration (i.e. for each
                    # position and velocity we give what would be the
                    # optimal acceleration, even though the next position
                    # and velocity won't reflect this decision, but the
                    # one taken by the learned policy)

                    xDAG, yDAG, SDAG = self.fixedBatchDAGger(initPosTrain,
                                                             initVelTrain)
                        
                    xTrain = np.concatenate((xTrain, xDAG), axis = 0)
                    Strain = np.concatenate((Strain, SDAG), axis = 0)
                    yTrain = np.concatenate((yTrain, yDAG), axis = 0)
                    initVelTrain = np.tile(initVelTrain, (2,1,1))
                #\\\
                #\\\ Finished DAGGER
                #\\\\\\\\\\\\\\\\\\\

                # Now that we have our dataset, move it to tensor and device
                # so we can use it
                xTrain = torch.tensor(xTrain, device = thisDevice)
                Strain = torch.tensor(Strain, device = thisDevice)
                yTrain = torch.tensor(yTrain, device = thisDevice)
                initVelTrain = torch.tensor(initVelTrain, device = thisDevice)

                # Start measuring time
                startTime = datetime.datetime.now()

                # Reset gradients
                thisArchit.zero_grad()

                # Obtain the output of the GNN
                yHatTrain = thisArchit(xTrain, Strain)

                # Compute loss
                lossValueTrain = thisLoss(yHatTrain, yTrain)

                # Compute gradients
                lossValueTrain.backward()

                # Optimize
                thisOptim.step()

                # Finish measuring time
                endTime = datetime.datetime.now()

                timeElapsed = abs(endTime - startTime).total_seconds()

                # Logging values
                if doLogging:
                    lossTrainTB = lossValueTrain.item()
                # Save values
                if doSaveVars:
                    lossTrain += [lossValueTrain.item()]
                    timeTrain += [timeElapsed]

                # Print:
                if doPrint and printInterval > 0:
                    if (epoch * nBatches + batch) % printInterval == 0:
                        print("\t(E: %2d, B: %3d) %7.4f - %6.4fs" % (
                                epoch+1, batch+1,
                                lossValueTrain.item(), timeElapsed),
                            end = ' ')
                        if graphNo > -1:
                            print("[%d" % graphNo, end = '')
                            if realizationNo > -1:
                                print("/%d" % realizationNo,
                                      end = '')
                            print("]", end = '')
                        print("")


                # Delete variables to free space in CUDA memory
                del xTrain
                del Strain
                del yTrain
                del initVelTrain
                del lossValueTrain

                #\\\\\\\
                #\\\ TB LOGGING (for each batch)
                #\\\\\\\

                if doLogging:
                    logger.scalar_summary(mode = 'Training',
                                          epoch = epoch * nBatches + batch,
                                          **{'lossTrain': lossTrainTB})

                #\\\\\\\
                #\\\ VALIDATION
                #\\\\\\\

                if (epoch * nBatches + batch) % validationInterval == 0:
                    
                    # Start measuring time
                    startTime = datetime.datetime.now()
                    
                    # Create trajectories
                    
                    # Initial data
                    initPosValid = self.data.getData('initPos','valid')
                    initVelValid = self.data.getData('initVel','valid')
                    
                    # Compute trajectories
                    _, velTestValid, _, _, _ = self.data.computeTrajectory(
                            initPosValid, initVelValid, self.data.duration,
                            archit = thisArchit, doPrint = False)
                    
                    # Compute evaluation
                    accValid = self.data.evaluate(vel = velTestValid)

                    # Finish measuring time
                    endTime = datetime.datetime.now()

                    timeElapsed = abs(endTime - startTime).total_seconds()

                    # Logging values
                    if doLogging:
                        evalValidTB = accValid
                    # Save values
                    if doSaveVars:
                        evalValid += [accValid]
                        timeValid += [timeElapsed]

                    # Print:
                    if doPrint:
                        print("\t(E: %2d, B: %3d) %8.4f - %6.4fs" % (
                                epoch+1, batch+1,
                                accValid, 
                                timeElapsed), end = ' ')
                        print("[VALIDATION", end = '')
                        if graphNo > -1:
                            print(".%d" % graphNo, end = '')
                            if realizationNo > -1:
                                print("/%d" % realizationNo, end = '')
                        print(" (%s)]" % self.model.name)

                    if doLogging:
                        logger.scalar_summary(mode = 'Validation',
                                          epoch = epoch * nBatches + batch,
                                          **{'evalValid': evalValidTB})

                    # No previous best option, so let's record the first trial
                    # as the best option
                    if epoch == 0 and batch == 0:
                        bestScore = accValid
                        bestEpoch, bestBatch = epoch, batch
                        # Save this model as the best (so far)
                        self.model.save(label = 'Best')
                        # Start the counter
                        if doEarlyStopping:
                            initialBest = True
                    else:
                        thisValidScore = accValid
                        if thisValidScore < bestScore:
                            bestScore = thisValidScore
                            bestEpoch, bestBatch = epoch, batch
                            if doPrint:
                                print("\t=> New best achieved: %.4f" % \
                                          (bestScore))
                            self.model.save(label = 'Best')
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

                    # Delete variables to free space in CUDA memory
                    del initVelValid
                    del initPosValid

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
        self.model.save(label = 'Last')

        #################
        # TRAINING OVER #
        #################

        if doSaveVars:
            # We convert the lists into np.arrays
            lossTrain = np.array(lossTrain)
            evalValid = np.array(evalValid)
            # And we would like to save all the relevant information from
            # training
            trainVars = {'nEpochs': nEpochs,
                     'nBatches': nBatches,
                     'validationInterval': validationInterval,
                     'batchSize': np.array(batchSize),
                     'batchIndex': np.array(batchIndex),
                     'bestBatch': bestBatch,
                     'bestEpoch': bestEpoch,
                     'bestScore': bestScore,
                     'lossTrain': lossTrain,
                     'timeTrain': timeTrain,
                     'evalValid': evalValid,
                     'timeValid': timeValid
                     }
            saveDirVars = os.path.join(self.model.saveDir, 'trainVars')
            if not os.path.exists(saveDirVars):
                os.makedirs(saveDirVars)
            pathToFile = os.path.join(saveDirVars,self.model.name + 'trainVars.pkl')
            with open(pathToFile, 'wb') as trainVarsFile:
                pickle.dump(trainVars, trainVarsFile)

        # Now, if we didn't do any training (i.e. nEpochs = 0), then the last is
        # also the best.
        if nEpochs == 0:
            self.model.save(label = 'Best')
            self.model.save(label = 'Last')
            if doPrint:
                print("\nWARNING: No training. Best and Last models are the same.\n")

        # After training is done, reload best model before proceeding to
        # evaluation:
        self.model.load(label = 'Best')

        #\\\ Print out best:
        if doPrint and nEpochs > 0:
            print("\t=> Best validation achieved (E: %d, B: %d): %.4f" % (
                    bestEpoch + 1, bestBatch + 1, bestScore))

        return trainVars
    
    def randomEpochDAGger(self, epoch, xTrainOrig, yTrainOrig, StrainOrig,
                          initPosTrainAll, initVelTrainAll):
        
        # The 'randomEpoch' option forms a new training set for each
        # epoch consisting, with probability probExpert, of samples
        # of the original dataset (optimal trajectories) and with
        # probability 1-probExpert, with trajectories following the
        # latest trained dataset.
        
        assert 'probExpert' in self.trainingOptions.kwargs()
        probExpert = self.trainingOptions['probExpert']
        nTrain = xTrainOrig.shape[0]

        # Compute the prob expert
        chooseExpertProb = np.max((probExpert ** epoch, 0.5))

        # What we will pass to the actual training epoch are:
        # xTrain, Strain and yTrain for computation
        xDAG = np.zeros(xTrainOrig.shape)
        yDAG = np.zeros(yTrainOrig.shape)
        SDAG = np.zeros(StrainOrig.shape)
        # initVelTrain is needed for evaluation, but doesn't change

        # For each sample, choose whether we keep the optimal
        # trajectory or we add the learned trajectory
        for s in range(nTrain):

            if np.random.binomial(1, chooseExpertProb) == 1:

                # If we choose the expert, we just get the values of
                # the optimal trajectory

                xDAG[s] = xTrainOrig[s]
                yDAG[s] = yTrainOrig[s]
                SDAG[s] = StrainOrig[s]

            else:

                # If not, we compute a new trajectory based on the
                # given architecture
                posDAG, velDAG, _, _, _ = self.data.computeTrajectory(
                    initPosTrainAll[s:s+1], initVelTrainAll[s:s+1],
                    self.data.duration, archit = self.model.archit,
                    doPrint = False)

                # Now that we have the position and velocity trajectory
                # that we would get based on the learned controller,
                # we need to compute what the optimal acceleration
                # would actually be in each case.
                # And since this could be a large trajectory, we need
                # to split it based on how many samples

                maxTimeSamples = 200

                if posDAG.shape[1] > maxTimeSamples:

                    # Create the space
                    yDAGaux = np.zeros((1, # batchSize
                                        posDAG.shape[1], # tSamples
                                        2,
                                        posDAG.shape[3])) # nAgents

                    for t in range(posDAG.shape[1]):

                        # Compute the expert on the corresponding
                        # trajectory
                        #   First, we need the difference in positions
                        ijDiffPos, ijDistSq = \
                               self.data.computeDifferences(posDAG[:,t,:,:])
                        #   And in velocities
                        ijDiffVel, _ = \
                               self.data.computeDifferences(velDAG[:,t,:,:])
                        #   Now, the second term (the one that depends
                        #   on the positions) only needs to be computed
                        #   for nodes thatare within repel distance, so
                        #   let's compute a mask to find these nodes.
                        repelMask = (ijDistSq < (self.data.repelDist ** 2))\
                                               .astype(ijDiffPos.dtype)
                        #   Apply this mask to the position difference
                        #   (we need not apply it to the square
                        #   differences since these will be multiplied
                        #   by the position differences which already
                        #   will be zero)
                        #   Note that we need to add the dimension of axis
                        #   to properly multiply it
                        ijDiffPos = ijDiffPos *\
                                            np.expand_dims(repelMask,1)
                        #   Invert the tensor elementwise (avoiding the
                        #   zeros)
                        ijDistSqInv = invertTensorEW(ijDistSq)
                        #   Add an extra dimension, also across the
                        #   axis
                        ijDistSqInv = np.expand_dims(ijDistSqInv, 1)
                        #   Compute the optimal solution
                        thisAccel = -np.sum(ijDiffVel, axis = 3) \
                                + 2 * np.sum(ijDiffPos * \
                                      (ijDistSqInv ** 2 + ijDistSqInv),
                                             axis = 3)
                        # And cap it
                        thisAccel[thisAccel > self.data.accelMax] = \
                                                          self.data.accelMax
                        thisAccel[thisAccel < -self.data.accelMax] = \
                                                         -self.data.accelMax

                        # Store it
                        yDAGaux[:,t,:,:] = thisAccel

                else:
                    # Compute the expert on the corresponding
                    # trajectory
                    #   First, we need the difference in positions
                    ijDiffPos,ijDistSq=self.data.computeDifferences(posDAG)
                    #   And in velocities
                    ijDiffVel, _ = self.data.computeDifferences(velDAG)
                    #   Now, the second term (the one that depends on
                    #   the positions) only needs to be computed for
                    #   nodes that are within repel distance, so let's
                    #   compute a mask to find these nodes.
                    repelMask = (ijDistSq < (self.data.repelDist ** 2))\
                                               .astype(ijDiffPos.dtype)
                    #   Apply this mask to the position difference (we
                    #   need not apply it to the square differences,
                    #   since these will be multiplied by the position
                    #   differences, which already will be zero)
                    #   Note that we need to add the dimension of axis
                    #   to properly multiply it
                    ijDiffPos = ijDiffPos * np.expand_dims(repelMask,2)
                    #   Invert the tensor elementwise (avoiding the
                    #   zeros)
                    ijDistSqInv = invertTensorEW(ijDistSq)
                    #   Add an extra dimension, also across the axis
                    ijDistSqInv = np.expand_dims(ijDistSqInv, 2)
                    #   Compute the optimal solution
                    yDAGaux = -np.sum(ijDiffVel, axis = 4) \
                            + 2 * np.sum(ijDiffPos * \
                                          (ijDistSqInv**2+ijDistSqInv),
                                         axis = 4)
                    # And cap it
                    yDAGaux[yDAGaux > self.data.accelMax] = self.data.accelMax
                    yDAGaux[yDAGaux < -self.data.accelMax] = -self.data.accelMax

                # Finally, compute the corresponding graph of states
                # (pos) visited by the policy
                SDAGaux = self.data.computeCommunicationGraph(
                        posDAG, self.data.commRadius, True, doPrint = False)
                xDAGaux = self.data.computeStates(posDAG, velDAG, SDAGaux,
                                             doPrint = False)

                # And save them
                xDAG[s] = xDAGaux[0]
                yDAG[s] = yDAGaux[0]
                SDAG[s] = SDAGaux[0]
                
        # And now that we have created the DAGger alternatives, we
        # just need to consider them as the basic training variables
        return xDAG, yDAG, SDAG
    
    def replaceTimeBatchDAGger(self, epoch, xTrainAll, yTrainAll, StrainAll,
                               initPosTrainAll, initVelTrainAll, nReplace = 10):
        
        # The option 'replaceTimeBatch' creates a fixed number of
        # new trajectories following randomly at each time step
        # either the optimal control or the learned control
        # Then, replaces this fixed number of new trajectores into
        # the training set (then these might, or might not get
        # selected by the next batch)
        
        assert 'probExpert' in self.trainingOptions.kwargs()
        probExpert = self.trainingOptions['probExpert']
        nTrain = xTrainAll.shape[0]

        if nReplace > nTrain:
            nReplace = nTrain

        # Select the indices of the samples to replace
        replaceIndices = np.random.permutation(nTrain)[0:nReplace]

        # Get the corresponding initial velocities and positions
        initPosTrainThis = initPosTrainAll[replaceIndices]
        initVelTrainThis = initVelTrainAll[replaceIndices]

        # Save the resulting trajectories
        xDAG = np.zeros((nReplace,
                         xTrainAll.shape[1],
                         6,
                         xTrainAll.shape[3]))
        yDAG = np.zeros((nReplace,
                         yTrainAll.shape[1],
                         2,
                         yTrainAll.shape[3]))
        SDAG = np.zeros((nReplace,
                         StrainAll.shape[1],
                         StrainAll.shape[2],
                         StrainAll.shape[3]))
        posDAG = np.zeros(yDAG.shape)
        velDAG = np.zeros(yDAG.shape)

        # Initialize first elements
        posDAG[:,0,:,:] = initPosTrainThis
        velDAG[:,0,:,:] = initVelTrainThis
        SDAG[:,0,:,:] = StrainAll[replaceIndices,0]
        xDAG[:,0,:,:] = xTrainAll[replaceIndices,0]

        # Compute the prob expert
        chooseExpertProb = np.max((probExpert ** (epoch+1), 0.5))

        # Now, for each sample
        for s in range(nReplace):

            # For each time instant
            for t in range(1,xTrainAll.shape[1]):

                # Decide whether we apply the learned or the
                # optimal controller
                if np.random.binomial(1, chooseExpertProb) == 1:

                    # Compute the optimal acceleration
                    ijDiffPos, ijDistSq = \
                     self.data.computeDifferences(posDAG[s:s+1,t-1,:,:])
                    ijDiffVel, _ = \
                     self.data.computeDifferences(velDAG[s:s+1,t-1,:,:])
                    repelMask = (ijDistSq < (self.data.repelDist ** 2))\
                                           .astype(ijDiffPos.dtype)
                    ijDiffPos = ijDiffPos *\
                                        np.expand_dims(repelMask,1)
                    ijDistSqInv = invertTensorEW(ijDistSq)
                    ijDistSqInv = np.expand_dims(ijDistSqInv, 1)
                    thisAccel = -np.sum(ijDiffVel, axis = 3) \
                            + 2 * np.sum(ijDiffPos * \
                                  (ijDistSqInv ** 2 + ijDistSqInv),
                                         axis = 3)
                else:

                    # Compute the learned acceleration
                    #   Add the sample dimension
                    xThis = np.expand_dims(xDAG[s,0:t,:,:], 0)
                    Sthis = np.expand_dims(SDAG[s,0:t,:,:], 0)
                    #   Convert to tensor
                    xThis = torch.tensor(xThis, device=self.model.device)
                    Sthis = torch.tensor(Sthis, device=self.model.device)
                    #   Compute the acceleration
                    with torch.no_grad():
                        thisAccel = self.model.archit(xThis, Sthis)
                    #   Get only the last acceleration
                    thisAccel = thisAccel.cpu().numpy()[:,-1,:,:]

                # Cap the acceleration
                thisAccel[thisAccel>self.data.accelMax]=self.data.accelMax
                thisAccel[thisAccel<-self.data.accelMax]=-self.data.accelMax
                # Save it
                yDAG[s,t-1,:,:] = thisAccel.squeeze(0)

                # Update the position and velocity
                velDAG[s,t,:,:] = \
                               yDAG[s,t-1,:,:] * self.data.samplingTime\
                                                + velDAG[s,t-1,:,:]
                posDAG[s,t,:,:] = \
                             velDAG[s,t-1,:,:] * self.data.samplingTime\
                                                + posDAG[s,t-1,:,:]
                # Update the state and the graph
                thisGraph = self.data.computeCommunicationGraph(
                    posDAG[s:s+1,t:t+1,:,:], self.data.commRadius,
                    True, doPrint = False)
                SDAG[s,t,:,:] = thisGraph.squeeze(1).squeeze(0)
                thisState = self.data.computeStates(
                    posDAG[s:s+1,t:t+1,:,:],
                    velDAG[s:s+1,t:t+1,:,:],
                    SDAG[s:s+1,t:t+1,:,:],
                    doPrint = False)
                xDAG[s,t,:,:] = thisState.squeeze(1).squeeze(0)

            # And now compute the last acceleration step

            if np.random.binomial(1, chooseExpertProb) == 1:

                # Compute the optimal acceleration
                ijDiffPos, ijDistSq = \
                   self.data.computeDifferences(posDAG[s:s+1,-1,:,:])
                ijDiffVel, _ = \
                   self.data.computeDifferences(velDAG[s:s+1,-1,:,:])
                repelMask = (ijDistSq < (self.data.repelDist ** 2))\
                                       .astype(ijDiffPos.dtype)
                ijDiffPos = ijDiffPos *\
                                    np.expand_dims(repelMask,1)
                ijDistSqInv = invertTensorEW(ijDistSq)
                ijDistSqInv = np.expand_dims(ijDistSqInv, 1)
                thisAccel = -np.sum(ijDiffVel, axis = 3) \
                        + 2 * np.sum(ijDiffPos * \
                              (ijDistSqInv ** 2 + ijDistSqInv),
                                     axis = 3)
            else:

                # Compute the learned acceleration
                #   Add the sample dimension
                xThis = np.expand_dims(xDAG[s], 0)
                Sthis = np.expand_dims(SDAG[s], 0)
                #   Convert to tensor
                xThis = torch.tensor(xThis, device=self.model.device)
                Sthis = torch.tensor(Sthis, device=self.model.device)
                #   Compute the acceleration
                with torch.no_grad():
                    thisAccel = self.model.archit(xThis, Sthis)
                #   Get only the last acceleration
                thisAccel = thisAccel.cpu().numpy()[:,-1,:,:]

            # Cap the acceleration
            thisAccel[thisAccel>self.data.accelMax]=self.data.accelMax
            thisAccel[thisAccel<-self.data.accelMax]=-self.data.accelMax
            # Save it
            yDAG[s,-1,:,:] = thisAccel.squeeze(0)

        # And now that we have done this for all the samples in
        # the replacement set, just replace them
            
        xTrainAll[replaceIndices] = xDAG
        yTrainAll[replaceIndices] = yDAG
        StrainAll[replaceIndices] = SDAG
        
        return xTrainAll, yTrainAll, StrainAll
    
    def fixedBatchDAGger(self, initPosTrain, initVelTrain):
        
        # The 'fixedBatch' option, doubles the batch samples
        # by considering the same initial velocities and
        # positions, a trajectory given by the latest trained
        # architecture, and the corresponding correction
        # given by the optimal acceleration (i.e. for each
        # position and velocity we give what would be the
        # optimal acceleration, even though the next position
        # and velocity won't reflect this decision, but the
        # one taken by the learned policy)
        
        # Note that there's no point on doing it randomly here,
        # since the optimal trajectory is already considered in
        # the batch anyways.

        #\\\\\\\\\\\\\\\\
        #\\\ Start DAGGER

        # Always apply DAGger on the trained policy
        posPol, velPol, _, _, _ = \
            self.data.computeTrajectory(initPosTrain,
                                   initVelTrain,
                                   self.data.duration,
                                   archit = self.model.archit,
                                   doPrint = False)

        # Compute the optimal acceleration on the trajectory given
        # by the trained policy

        maxTimeSamples = 200

        if posPol.shape[1] > maxTimeSamples:

            # Create the space to store this
            yDAG = np.zeros(posPol.shape)

            for t in range(posPol.shape[1]):

                # Compute the expert on the corresponding trajectory
                #   First, we need the difference in positions
                ijDiffPos, ijDistSq = \
                           self.data.computeDifferences(posPol[:,t,:,:])
                #   And in velocities
                ijDiffVel, _ = \
                           self.data.computeDifferences(velPol[:,t,:,:])
                #   Now, the second term (the one that depends on
                #   the positions) only needs to be computed for
                #   nodes thatare within repel distance, so let's
                #   compute a mask to find these nodes.
                repelMask = (ijDistSq < (self.data.repelDist ** 2))\
                                           .astype(ijDiffPos.dtype)
                #   Apply this mask to the position difference (we
                #   need not apply it to the square differences,
                #   since these will be multiplied by the position
                #   differences which already will be zero)
                #   Note that we need to add the dimension of axis
                #   to properly multiply it
                ijDiffPos = ijDiffPos * np.expand_dims(repelMask,1)
                #   Invert the tensor elementwise (avoiding the
                #   zeros)
                ijDistSqInv = invertTensorEW(ijDistSq)
                #   Add an extra dimension, also across the axis
                ijDistSqInv = np.expand_dims(ijDistSqInv, 1)
                #   Compute the optimal solution
                thisAccel = -np.sum(ijDiffVel, axis = 3) \
                        + 2 * np.sum(ijDiffPos * \
                                  (ijDistSqInv ** 2 + ijDistSqInv),
                                     axis = 3)
                # And cap it
                thisAccel[thisAccel>self.data.accelMax]=self.data.accelMax
                thisAccel[thisAccel<-self.data.accelMax]=-self.data.accelMax

                # Store it
                yDAG[:,t,:,:] = thisAccel

        else:
            # Compute the expert on the corresponding trajectory
            #   First, we need the difference in positions
            ijDiffPos, ijDistSq = self.data.computeDifferences(posPol)
            #   And in velocities
            ijDiffVel, _ = self.data.computeDifferences(velPol)
            #   Now, the second term (the one that depends on the
            #   positions) only needs to be computed for nodes that
            #   are within repel distance, so let's compute a mask
            #   to find these nodes.
            repelMask = (ijDistSq < (self.data.repelDist ** 2))\
                                           .astype(ijDiffPos.dtype)
            #   Apply this mask to the position difference (we need
            #   not apply it to the square differences, since these
            #   will be multiplied by the position differences,
            #   which already will be zero)
            #   Note that we need to add the dimension of axis to
            #   properly multiply it
            ijDiffPos = ijDiffPos * np.expand_dims(repelMask, 2)
            #   Invert the tensor elementwise (avoiding the zeros)
            ijDistSqInv = invertTensorEW(ijDistSq)
            #   Add an extra dimension, also across the axis
            ijDistSqInv = np.expand_dims(ijDistSqInv, 2)
            #   Compute the optimal solution
            yDAG = -np.sum(ijDiffVel, axis = 4) \
                    + 2 * np.sum(ijDiffPos * \
                                  (ijDistSqInv ** 2 + ijDistSqInv),
                                 axis = 4)
            # And cap it
            yDAG[yDAG > self.data.accelMax] = self.data.accelMax
            yDAG[yDAG < -self.data.accelMax] = -self.data.accelMax

        # Finally, compute the corresponding graph of states
        # (pos) visited by the policy
        graphDAG = self.data.computeCommunicationGraph(posPol,
                                                       self.data.commRadius,
                                                       True,
                                                       doPrint = False)
        xDAG = self.data.computeStates(posPol, velPol, graphDAG,
                                       doPrint = False)

        # Add it to the existing batch
        
        return xDAG, yDAG, graphDAG