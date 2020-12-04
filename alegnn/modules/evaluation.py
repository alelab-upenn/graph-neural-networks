# 2020/02/25~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
evaluation.py Evaluation Module

Methods for evaluating the models.

evaluate: evaluate a model
evaluateSingleNode: evaluate a model that has a single node forward
evaluateFlocking: evaluate a model using the flocking cost
"""

import os
import torch
import pickle

def evaluate(model, data, **kwargs):
    """
    evaluate: evaluate a model using classification error
    
    Input:
        model (model class): class from Modules.model
        data (data class): a data class from the Utils.dataTools; it needs to
            have a getSamples method and an evaluate method.
        doPrint (optional, bool): if True prints results
    
    Output:
        evalVars (dict): 'errorBest' contains the error rate for the best
            model, and 'errorLast' contains the error rate for the last model
    """

    # Get the device we're working on
    device = model.device
    
    if 'doSaveVars' in kwargs.keys():
        doSaveVars = kwargs['doSaveVars']
    else:
        doSaveVars = True

    ########
    # DATA #
    ########

    xTest, yTest = data.getSamples('test')
    xTest = xTest.to(device)
    yTest = yTest.to(device)

    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit(xTest)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costBest = data.evaluate(yHatTest, yTest)

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit(xTest)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costLast = data.evaluate(yHatTest, yTest)

    evalVars = {}
    evalVars['costBest'] = costBest.item()
    evalVars['costLast'] = costLast.item()
    
    if doSaveVars:
        saveDirVars = os.path.join(model.saveDir, 'evalVars')
        if not os.path.exists(saveDirVars):
            os.makedirs(saveDirVars)
        pathToFile = os.path.join(saveDirVars, model.name + 'evalVars.pkl')
        with open(pathToFile, 'wb') as evalVarsFile:
            pickle.dump(evalVars, evalVarsFile)

    return evalVars

def evaluateSingleNode(model, data, **kwargs):
    """
    evaluateSingleNode: evaluate a model that has a single node forward
    
    Input:
        model (model class): class from Modules.model, needs to have a 
            'singleNodeForward' method
        data (data class): a data class from the Utils.dataTools; it needs to
            have a getSamples method and an evaluate method and it also needs to
            have a 'getLabelID' method
        doPrint (optional, bool): if True prints results
    
    Output:
        evalVars (dict): 'errorBest' contains the error rate for the best
            model, and 'errorLast' contains the error rate for the last model
    """
    
    assert 'singleNodeForward' in dir(model.archit)
    assert 'getLabelID' in dir(data)

    # Get the device we're working on
    device = model.device
    
    if 'doSaveVars' in kwargs.keys():
        doSaveVars = kwargs['doSaveVars']
    else:
        doSaveVars = True

    ########
    # DATA #
    ########

    xTest, yTest = data.getSamples('test')
    xTest = xTest.to(device)
    yTest = yTest.to(device)
    targetIDs = data.getLabelID('test')

    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit.singleNodeForward(xTest, targetIDs)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costBest = data.evaluate(yHatTest, yTest)

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit.singleNodeForward(xTest, targetIDs)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costLast = data.evaluate(yHatTest, yTest)

    evalVars = {}
    evalVars['costBest'] = costBest.item()
    evalVars['costLast'] = costLast.item()
    
    if doSaveVars:
        saveDirVars = os.path.join(model.saveDir, 'evalVars')
        if not os.path.exists(saveDirVars):
            os.makedirs(saveDirVars)
        pathToFile = os.path.join(saveDirVars, model.name + 'evalVars.pkl')
        with open(pathToFile, 'wb') as evalVarsFile:
            pickle.dump(evalVars, evalVarsFile)

    return evalVars

def evaluateFlocking(model, data, **kwargs):
    """
    evaluateClassif: evaluate a model using the flocking cost of velocity 
        variacne of the team
    
    Input:
        model (model class): class from Modules.model
        data (data class): the data class that generates the flocking data
        doPrint (optional; bool, default: True): if True prints results
        nVideos (optional; int, default: 3): number of videos to save
        graphNo (optional): identify the run with a number
        realizationNo (optional): identify the run with another number
    
    Output:
        evalVars (dict):
            'costBestFull': cost of the best model over the full trajectory
            'costBestEnd': cost of the best model at the end of the trajectory
            'costLastFull': cost of the last model over the full trajectory
            'costLastEnd': cost of the last model at the end of the trajectory
    """
    
    if 'doPrint' in kwargs.keys():
        doPrint = kwargs['doPrint']
    else:
        doPrint = True
        
    if 'nVideos' in kwargs.keys():
        nVideos = kwargs['nVideos']
    else:
        nVideos = 3
        
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

    #\\\\\\\\\\\\\\\\\\\\
    #\\\ TRAJECTORIES \\\
    #\\\\\\\\\\\\\\\\\\\\

    ########
    # DATA #
    ########

    # Initial data
    initPosTest = data.getData('initPos', 'test')
    initVelTest = data.getData('initVel', 'test')

    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    if doPrint:
        print("\tComputing learned trajectory for best model...",
              end = ' ', flush = True)

    posTestBest, \
    velTestBest, \
    accelTestBest, \
    stateTestBest, \
    commGraphTestBest = \
        data.computeTrajectory(initPosTest, initVelTest, data.duration,
                               archit = model.archit)

    if doPrint:
        print("OK")

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    if doPrint:
        print("\tComputing learned trajectory for last model...",
              end = ' ', flush = True)

    posTestLast, \
    velTestLast, \
    accelTestLast, \
    stateTestLast, \
    commGraphTestLast = \
        data.computeTrajectory(initPosTest, initVelTest, data.duration,
                               archit = model.archit)

    if doPrint:
        print("OK")

    ###########
    # PREVIEW #
    ###########

    learnedTrajectoriesDir = os.path.join(model.saveDir,
                                          'learnedTrajectories')
    
    if not os.path.exists(learnedTrajectoriesDir):
        os.mkdir(learnedTrajectoriesDir)
    
    if graphNo > -1:
        learnedTrajectoriesDir = os.path.join(learnedTrajectoriesDir,
                                              '%03d' % graphNo)
        if not os.path.exists(learnedTrajectoriesDir):
            os.mkdir(learnedTrajectoriesDir)
    if realizationNo > -1:
        learnedTrajectoriesDir = os.path.join(learnedTrajectoriesDir,
                                              '%03d' % realizationNo)
        if not os.path.exists(learnedTrajectoriesDir):
            os.mkdir(learnedTrajectoriesDir)

    learnedTrajectoriesDir = os.path.join(learnedTrajectoriesDir, model.name)

    if not os.path.exists(learnedTrajectoriesDir):
        os.mkdir(learnedTrajectoriesDir)

    if doPrint:
        print("\tPreview data...",
              end = ' ', flush = True)

    data.saveVideo(os.path.join(learnedTrajectoriesDir,'Best'),
                   posTestBest,
                   nVideos,
                   commGraph = commGraphTestBest,
                   vel = velTestBest,
                   videoSpeed = 0.5,
                   doPrint = False)

    data.saveVideo(os.path.join(learnedTrajectoriesDir,'Last'),
                   posTestLast,
                   nVideos,
                   commGraph = commGraphTestLast,
                   vel = velTestLast,
                   videoSpeed = 0.5,
                   doPrint = False)

    if doPrint:
        print("OK", flush = True)

    #\\\\\\\\\\\\\\\\\\
    #\\\ EVALUATION \\\
    #\\\\\\\\\\\\\\\\\\
        
    evalVars = {}
    evalVars['costBestFull'] = data.evaluate(vel = velTestBest)
    evalVars['costBestEnd'] = data.evaluate(vel = velTestBest[:,-1:,:,:])
    evalVars['costLastFull'] = data.evaluate(vel = velTestLast)
    evalVars['costLastEnd'] = data.evaluate(vel = velTestLast[:,-1:,:,:])

    return evalVars