# 2021/03/04~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
# Kate Tolstaya, eig@seas.upenn.edu
"""
dataTools.py Data management module

Functions:
    
normalizeData: normalize data along a specified axis
changeDataType: change data type of data

Classes (datasets):

FacebookEgo (class): loads the Facebook adjacency matrix of EgoNets
SourceLocalization (class): creates the datasets for a source localization 
    problem
Authorship (class): loads and splits the dataset for the authorship attribution
    problem
MovieLens (class): Loads and handles handles the MovieLens-100k dataset

Flocking (class): creates trajectories for the problem of flocking

TwentyNews (class): handles the 20NEWS dataset

Epidemics (class): loads the edge list of the friendship network of the high 
    school in Marseille and generates the epidemic spread data based on the SIR 
    model
"""

import os
import pickle
import hdf5storage # This is required to import old Matlab(R) files.
import urllib.request # To download from the internet
import zipfile # To handle zip files
import gzip # To handle gz files
import shutil # Command line utilities
import matplotlib
import csv
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import numpy as np
import torch

import alegnn.utils.graphTools as graph

zeroTolerance = 1e-9 # Values below this number are considered zero.

def normalizeData(x, ax):
    """
    normalizeData(x, ax): normalize data x (subtract mean and divide by standard 
    deviation) along the specified axis ax
    """
    
    thisShape = x.shape # get the shape
    assert ax < len(thisShape) # check that the axis that we want to normalize
        # is there
    dataType = type(x) # get data type so that we don't have to convert

    if 'numpy' in repr(dataType):

        # Compute the statistics
        xMean = np.mean(x, axis = ax)
        xDev = np.std(x, axis = ax)
        # Add back the dimension we just took out
        xMean = np.expand_dims(xMean, ax)
        xDev = np.expand_dims(xDev, ax)

    elif 'torch' in repr(dataType):

        # Compute the statistics
        xMean = torch.mean(x, dim = ax)
        xDev = torch.std(x, dim = ax)
        # Add back the dimension we just took out
        xMean = xMean.unsqueeze(ax)
        xDev = xDev.unsqueeze(ax)

    # Subtract mean and divide by standard deviation
    x = (x - xMean) / xDev

    return x

def changeDataType(x, dataType):
    """
    changeDataType(x, dataType): change the dataType of variable x into dataType
    """
    
    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.
    
    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.
    
    # If we can't recognize the type, we just make everything numpy.
    
    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype
    
    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype = dataType)
            
    # This only converts between numpy and torch. Any other thing is ignored
    return x

def invertTensorEW(x):
    
    # Elementwise inversion of a tensor where the 0 elements are kept as zero.
    # Warning: Creates a copy of the tensor
    xInv = x.copy() # Copy the matrix to invert
    # Replace zeros for ones.
    xInv[x < zeroTolerance] = 1. # Replace zeros for ones
    xInv = 1./xInv # Now we can invert safely
    xInv[x < zeroTolerance] = 0. # Put back the zeros
    
    return xInv

class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   getSamples(), expandDims(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    
    # All the signals are always assumed to be graph signals that are written
    #   nDataPoints (x nFeatures) x nNodes
    # If we have one feature, we have the expandDims() that adds a x1 so that
    # it can be readily processed by architectures/functions that always assume
    # a 3-dimensional signal.
    
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['targets'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['targets'] = None
        
    def getSamples(self, samplesType, *args):
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['targets']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # Select the corresponding samples
                xSelected = x[selectedIndices]
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                xSelected = x[args[0]]
                # And assign the labels
                y = y[args[0]]
                
            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(xSelected.shape) < len(x.shape):
                if 'torch' in self.dataType:
                    x = xSelected.unsqueeze(0)
                else:
                    x = np.expand_dims(xSelected, axis = 0)
            else:
                x = xSelected

        return x, y
    
    def expandDims(self):
        
        # For each data set partition
        for key in self.samples.keys():
            # If there's something in them
            if self.samples[key]['signals'] is not None:
                # And if it has only two dimensions
                #   (shape: nDataPoints x nNodes)
                if len(self.samples[key]['signals'].shape) == 2:
                    # Then add a third dimension in between so that it ends
                    # up with shape
                    #   nDataPoints x 1 x nNodes
                    # and it respects the 3-dimensional format that is taken
                    # by many of the processing functions
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(1)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 1)
                elif len(self.samples[key]['signals'].shape) == 3:
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(2)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 2)
        
    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        
        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers also after conversion. 
        # To do this we need to match the desired dataType to its int 
        # counterpart. Typical examples are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32
        
        targetType = str(self.samples['train']['targets'].dtype)
        if 'int' in targetType:
            if 'numpy' in repr(dataType):
                if '64' in targetType:
                    targetType = np.int64
                elif '32' in targetType:
                    targetType = np.int32
            elif 'torch' in repr(dataType):
                if '64' in targetType:
                    targetType = torch.int64
                elif '32' in targetType:
                    targetType = torch.int32
        else: # If there is no int, just stick with the given dataType
            targetType = dataType
        
        # Now that we have selected the dataType, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        for key in self.samples.keys():
            self.samples[key]['signals'] = changeDataType(
                                                   self.samples[key]['signals'],
                                                   dataType)
            self.samples[key]['targets'] = changeDataType(
                                                   self.samples[key]['targets'],
                                                   targetType)

        # Update attribute
        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if 'torch' in repr(self.dataType):
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

class _dataForClassification(_data):
    # Internal supraclass from which data classes inherit when they are used
    # for classification. This renders the .evaluate() method the same in all
    # cases (how many examples are incorrectly labeled) so justifies the use of
    # another internal class.
    
    def __init__(self):
        
        super().__init__()
    

    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim = 1)
            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            errorRate = totalErrors.type(self.dataType)/N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis = 1)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            errorRate = totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
        return errorRate
        
class FacebookEgo:
    """
    FacebookEgo: Loads the adjacency matrix of the Facebook Egonets available
        in https://snap.stanford.edu/data/ego-Facebook.html by
        J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego
        Networks. NIPS, 2012.
        
    Initialization:
        
    Input:
        dataDir (string): path for the directory in where to look for the data 
            (if the data is not found, it will be downloaded to this directory)
        use234 (bool): if True, load a smaller subnetwork of 234 users with two
            communities (one big, and one small)
            
    Methods:
        
    .loadData(filename, use234): load the data in self.dataDir/filename, if it
        does not exist, then download it and save it as filename in self.dataDir
        If use234 is True, load the 234-user subnetwork as well.
        
    adjacencyMatrix = .getAdjacencyMatrix([use234]): return the nNodes x nNodes
        np.array with the adjacency matrix. If use234 is True, then return the
        smaller nNodes = 234 user subnetwork (default: use234 = False).
    """
    
    def __init__(self, dataDir, use234 = False):
        
        # Dataset directory
        self.dataDir = dataDir
        # Empty attributes
        self.adjacencyMatrix = None
        self.adjacencyMatrix234 = None
        
        # Load data
        self.loadData('facebookEgo.pkl', use234)
        
    def loadData(self, filename, use234):
        # Check if the dataDir exists, and if not, create it
        if not os.path.exists(self.dataDir):
            os.makedirs(self.dataDir)
        # Create the filename to save/load
        datasetFilename = os.path.join(self.dataDir, filename)
        if use234:
            datasetFilename234 = os.path.join(self.dataDir,'facebookEgo234.pkl')
            if os.path.isfile(datasetFilename234):
                with open(datasetFilename234, 'rb') as datasetFile234:
                    datasetDict = pickle.load(datasetFile234)
                    self.adjacencyMatrix234 = datasetDict['adjacencyMatrix']
        # Check if the file does exist, load it
        if os.path.isfile(datasetFilename):
            # If it exists, load it
            with open(datasetFilename, 'rb') as datasetFile:
                datasetDict = pickle.load(datasetFile)
                # And save the corresponding variable
                self.adjacencyMatrix = datasetDict['adjacencyMatrix']
        else: # If it doesn't exist, load it
            # There could be three options here: that we have the raw data 
            # already there, that we have the zip file and need to unzip it,
            # or that we do not have nothing and we need to download it.
            existsRawData = \
                   os.path.isfile(os.path.join(self.dataDir,
                                               'facebook_combined.txt'))
           # And the zip file
            existsZipFile = os.path.isfile(os.path.join(
                                       self.dataDir,'facebook_combined.txt.gz'))
            if not existsRawData and not existsZipFile: # We have to download it
                fbURL='https://snap.stanford.edu/data/facebook_combined.txt.gz'
                urllib.request.urlretrieve(fbURL,
                                 filename = os.path.join(
                                       self.dataDir,'facebook_combined.txt.gz'))
                existsZipFile = True
            if not existsRawData and existsZipFile: # Unzip it
                zipFile = os.path.join(self.dataDir, 'facebook_combined.txt.gz')
                txtFile = os.path.join(self.dataDir, 'facebook_combined.txt')
                with gzip.open(zipFile, 'rb') as f_in:
                    with open(txtFile, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            # Now that we have the data, we can get their filenames
            rawDataFilename = os.path.join(self.dataDir,'facebook_combined.txt')
            assert os.path.isfile(rawDataFilename)
            # And we can load it and store it.
            adjacencyMatrix = np.empty([0, 0]) # Start with an empty matrix and
            # then we slowly add the number of nodes, which we do not assume
            # to be known beforehand.
            # Let's start with the data.
            # Open it.
            with open(rawDataFilename, 'r') as rawData:
                # The file consists of a succession of lines, each line
                # corresponds to an edge
                for dataLine in rawData:
                    # For each line, we split it in the different fields
                    dataLineSplit = dataLine.rstrip('\n').split(' ')
                    # Keep the ones we care about here
                    node_i = int(dataLineSplit[0])
                    node_j = int(dataLineSplit[1])
                    node_max = max(node_i, node_j) # Get the largest node
                    # Now we have to add this information to the adjacency 
                    # matrix.
                    #   We need to check whether we need to add more elements
                    if node_max+1 > max(adjacencyMatrix.shape):
                        colDiff = node_max+1 - adjacencyMatrix.shape[1]
                        zeroPadCols = np.zeros([adjacencyMatrix.shape[0],\
                                                colDiff])
                        adjacencyMatrix = np.concatenate((adjacencyMatrix,
                                                          zeroPadCols),
                                                         axis = 1)
                        rowDiff = node_max+1 - adjacencyMatrix.shape[0]
                        zeroPadRows = np.zeros([rowDiff,\
                                                adjacencyMatrix.shape[1]])
                        adjacencyMatrix = np.concatenate((adjacencyMatrix,
                                                          zeroPadRows),
                                                         axis = 0)
                    # Now that we have assured appropriate dimensions
                    adjacencyMatrix[node_i, node_j] = 1.
                    # And because it is undirected by construction
                    adjacencyMatrix[node_j, node_i] = 1.
            # Now that it is loaded, let's store it
            self.adjacencyMatrix = adjacencyMatrix
            # And save it in a pickle file for posterity
            with open(datasetFilename, 'wb') as datasetFile:
                pickle.dump(
                        {'adjacencyMatrix': self.adjacencyMatrix},
                        datasetFile
                        )
    
    def getAdjacencyMatrix(self, use234 = False):
        
        return self.adjacencyMatrix234 if use234 else self.adjacencyMatrix

class SourceLocalization(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        sourceNodes (list of int): list of indices of nodes to be used as
            sources of the diffusion process
        tMax (int): maximum diffusion time, if None, the maximum diffusion time
            is the size of the graph (default: None)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label
                
    .expandDims(): Adds the feature dimension to the graph signals (i.e. for
        graph signals of shape nSamples x nNodes, turns them into shape
        nSamples x 1 x nNodes, so that they can be handled by general graph
        signal processing techniques that take into account a feature dimension
        by default)

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    errorRate = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): unnormalized probability of each label (shape:
                nDataPoints x nClasses)
            y (dtype.array): correct labels (1-D binary vector, shape:
                nDataPoints)
            tol (float, default = 1e-9): numerical tolerance to consider two
                numbers to be equal
        Output:
            errorRate (float): proportion of incorrect labels

    """

    def __init__(self, G, nTrain, nValid, nTest, sourceNodes, tMax = None,
                 dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        # If no tMax is specified, set it the maximum possible.
        if tMax == None:
            tMax = G.N
        #\\\ Generate the samples
        # Get the largest eigenvalue of the weighted adjacency matrix
        EW, VW = graph.computeGFT(G.W, order = 'totalVariation')
        eMax = np.max(EW)
        # Normalize the matrix so that it doesn't explode
        Wnorm = G.W / eMax
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
        sampledSources = np.random.choice(sourceNodes, size = nTotal)
        # sample diffusion times
        sampledTimes = np.random.choice(tMax, size = nTotal)
        # Since the signals are generated as W^t * delta, this reduces to the
        # selection of a column of W^t (the column corresponding to the source
        # node). Therefore, we generate an array of size tMax x N x N with all
        # the powers of the matrix, and then we just simply select the
        # corresponding column for the corresponding time
        lastWt = np.eye(G.N, G.N)
        Wt = lastWt.reshape([1, G.N, G.N])
        for t in range(1,tMax):
            lastWt = lastWt @ Wnorm
            Wt = np.concatenate((Wt, lastWt.reshape([1, G.N, G.N])), axis = 0)
        x = Wt[sampledTimes, :, sampledSources]
        # Now, we have the signals and the labels
        signals = x # nTotal x N
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.
        nodesToLabels = {}
        for it in range(len(sourceNodes)):
            nodesToLabels[sourceNodes[it]] = it
        labels = [nodesToLabels[x] for x in sampledSources] # nTotal
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['targets'] = np.array(labels[0:nTrain])
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['targets'] =np.array(labels[nTrain:nTrain+nValid])
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['targets'] =np.array(labels[nTrain+nValid:nTotal])
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
    
class Authorship(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem
        
    Credits for this dataset to Mark Eisen. Please, refer to this paper for
    details, and whenever using this dataset:
        S. Segarra, M. Eisen and A. Ribeiro, Authorship Attribution through
        Function Word Adjacency Networks, IEEE Trans. Signal Process., vol. 63,
        Issue 20, Oct 2015.
        
    Possible authors: 
        jacob 'abbott',         robert louis 'stevenson',   louisa may 'alcott',
        horatio 'alger',        james 'allen',              jane 'austen',
        emily 'bronte',         james 'cooper',             charles 'dickens', 
        hamlin 'garland',       nathaniel 'hawthorne',      henry 'james',
        herman 'melville',      'page',                     henry 'thoreau',
        mark 'twain',           arthur conan 'doyle',       washington 'irving',
        edgar allan 'poe',      sarah orne 'jewett',        edith 'wharton'

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        graphNormalizationType ('rows' or 'cols'): how to normalize the created
            graph from combining all the selected author WANs
        keepIsolatedNodes (bool): If False, get rid of isolated nodes
        forceUndirected (bool): If True, create an undirected graph
        forceConnected (bool): If True, ensure that the resulting graph is
            connected
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:
        
    .loadData(dataPath): load the data found in dataPath and store it in 
        attributes .authorData and .functionWords
        
    authorData = .getAuthorData(samplesType, selectData, [, optionalArguments])
    
        Input:
            samplesType (string): 'train', 'valid', 'test' or 'all' to determine
                from which dataset to get the raw author data from
            selectData (string): 'WAN' or 'wordFreq' to decide if we want to
                retrieve either the WAN of each excerpt or the word frequency
                count of each excerpt
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        
        Output:
            Either the WANs or the word frequency count of all the excerpts of
            the selected author
            
    .createGraph(): creates a graph from the WANs of the excerpt written by the
        selected author available in the training set. The fusion of this WANs
        is done in accordance with the input options following 
        graphTools.createGraph().
        The resulting adjacency matrix is stored.
        
    .getGraph(): fetches the stored adjacency matrix and returns it
    
    .getFunctionWords(): fetches the list of functional words. Returns a tuple
        where the first element correspond to all the functional words in use, 
        and the second element consists of all the functional words available.
        Obs.: When we created the graph, some of the functional words might have
        been dropped in order to make it connected, for example.

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label
                
    .expandDims(): Adds the feature dimension to the graph signals (i.e. for
        graph signals of shape nSamples x nNodes, turns them into shape
        nSamples x 1 x nNodes, so that they can be handled by general graph
        signal processing techniques that take into account a feature dimension
        by default)

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    errorRate = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            errorRate (float): proportion of incorrect labels

    """
    
    def __init__(self, authorName, ratioTrain, ratioValid, dataPath,
                 graphNormalizationType, keepIsolatedNodes,
                 forceUndirected, forceConnected,
                 dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.authorName = authorName
        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid
        self.dataPath = dataPath
        self.dataType = dataType
        self.device = device
        # Store characteristics of the graph to be created
        self.graphNormalizationType = graphNormalizationType
        self.keepIsolatedNodes = keepIsolatedNodes
        self.forceUndirected = forceUndirected
        self.forceConnected = forceConnected
        self.adjacencyMatrix = None
        # Other data to save
        self.authorData = None
        self.selectedAuthor = None
        self.allFunctionWords = None
        self.functionWords = None
        # Load data
        self.loadData(dataPath)
        # Check that the authorName is a valid name
        assert authorName in self.authorData.keys()
        # Get the selected author's data
        thisAuthorData = self.authorData[authorName].copy()
        nExcerpts = thisAuthorData['wordFreq'].shape[0] # Number of excerpts
            # by the selected author
        nTrainAuthor = int(round(ratioTrain * nExcerpts))
        nValidAuthor = int(round(ratioValid * nTrainAuthor))
        nTestAuthor = nExcerpts - nTrainAuthor
        nTrainAuthor = nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        self.nTrain = round(2 * nTrainAuthor)
        self.nValid = round(2 * nValidAuthor)
        self.nTest = round(2 * nTestAuthor)
        
        # Now, let's get the corresponding signals for the author
        xAuthor = thisAuthorData['wordFreq']
        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(nExcerpts)
        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor+nValidAuthor]
        randPermTest = randPerm[nTrainAuthor+nValidAuthor:nExcerpts]
        xAuthorTrain = xAuthor[randPermTrain, :]
        xAuthorValid = xAuthor[randPermValid, :]
        xAuthorTest = xAuthor[randPermTest, :]
        # And we will store this split
        self.selectedAuthor = {}
        # Copy all data
        self.selectedAuthor['all'] = thisAuthorData.copy()
        # Copy word frequencies
        self.selectedAuthor['train'] = {}
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
                              thisAuthorData['WAN'][randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
                              thisAuthorData['WAN'][randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
                               thisAuthorData['WAN'][randPermTest, :, :].copy()
        # Now we need to get an equal amount of works from the rest of the
        # authors.
        xRest = np.empty([0, xAuthorTrain.shape[1]]) # Create an empty matrix
        # to store all the works by the rest of the authors.
        # Now go author by author gathering all works
        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key is not authorName:
                thisAuthorTexts = self.authorData[key]['wordFreq']
                xRest = np.concatenate((xRest, thisAuthorTexts), axis = 0)
        # After obtaining all works, xRest is of shape nRestOfData x nWords
        # We now need to select at random from this other data, but only up
        # to nExcerpts. Therefore, we will randperm all the indices, but keep
        # only the first nExcerpts indices.
        randPerm = np.random.permutation(xRest.shape[0])
        randPerm = randPerm[0:nExcerpts] # nExcerpts x nWords
        # And now we should just get the appropriate number of texts from these
        # other authors.
        # Compute how many samples for each case
        nTrainRest = self.nTrain - nTrainAuthor
        nValidRest = self.nValid - nValidAuthor
        nTestRest = self.nTest - nTestAuthor
        # And obtain those
        xRestTrain = xRest[randPerm[0:nTrainRest], :]
        xRestValid = xRest[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        xRestTest = xRest[randPerm[nTrainRest+nValidRest:nExcerpts], :]
        # Now construct the signals and labels. Signals is just the 
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, xRestTrain), axis = 0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis = 0)
        xValid = np.concatenate((xAuthorValid, xRestValid), axis = 0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis = 0)
        xTest = np.concatenate((xAuthorTest, xRestTest), axis = 0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis = 0)
        # And assign them to the required attribute samples
        self.samples['train']['signals'] = xTrain
        self.samples['train']['targets'] = labelsTrain.astype(np.int)
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['targets'] = labelsValid.astype(np.int)
        self.samples['test']['signals'] = xTest
        self.samples['test']['targets'] = labelsTest.astype(np.int)
        # Create graph
        self.createGraph()
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def loadData(self, dataPath):
        # Load data (from a .mat file)
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old 
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {} # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0] 
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it] # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of 
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it] # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1) # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is 
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the 
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the 
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = [] # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.allFunctionWords = functionWords
        self.functionWords = functionWords.copy()
        
    def getAuthorData(self, samplesType, dataType, *args):
        # dataType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test' or samplesType == 'all'
        # Check that the dataType is either wordFreq or WAN
        assert dataType == 'WAN' or dataType == 'wordFreq'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.selectedAuthor[samplesType][dataType]
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # The reshape is to avoid squeezing if only one sample is
                # requested (because x can have two or three dimension, we
                # need to take a longer path here, so we will only do it
                # if args[0] is equal to 1.)
                if args[0] == 1:
                    newShape = [1]
                    newShape.extend(list(x.shape[1:]))
                    x = x[selectedIndices].reshape(newShape)
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                xNew = x[args[0]]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(xNew.shape) <= len(x.shape):
                    newShape = [1]
                    newShape.extend(list(x.shape[1:]))
                    x = xNew.reshape(newShape)

        return x
    
    def createGraph(self):
        
        # Save list of nodes to keep to later update the datasets with the
        # appropriate words
        nodesToKeep = []
        # Number of nodes (so far) = Number of functional words
        N = self.selectedAuthor['all']['wordFreq'].shape[1]
        # Create graph
        graphOptions = {}
        graphOptions['adjacencyMatrices'] = self.selectedAuthor['train']['WAN']
        graphOptions['nodeList'] = nodesToKeep
        graphOptions['aggregationType'] = 'sum'
        graphOptions['normalizationType'] = self.graphNormalizationType
        graphOptions['isolatedNodes'] = self.keepIsolatedNodes
        graphOptions['forceUndirected'] = self.forceUndirected
        graphOptions['forceConnected'] = self.forceConnected
        W = graph.createGraph('fuseEdges', N, graphOptions)
        # Obs.: We do not need to recall graphOptions['nodeList'] as nodesToKeep
        # since these are all passed as pointers that point to the same list, so
        # modifying graphOptions also modifies nodesToKeep.
        # Store adjacency matrix
        self.adjacencyMatrix = W.astype(np.float64)
        # Update data
        #   For each dataset split
        for key in self.samples.keys():
            #   Check the signals have been loaded
            if self.samples[key]['signals'] is not None:
                #   And check which is the dimension of the nodes (i.e. whether
                #   it was expanded or not, since we always need to keep the
                #   entries of the last dimension)
                if len(self.samples[key]['signals'].shape) == 2:
                    self.samples[key]['signals'] = \
                                   self.samples[key]['signals'][: , nodesToKeep]
                elif len(self.samples[key]['signals'].shape) == 2:
                    self.samples[key]['signals'] = \
                                   self.samples[key]['signals'][:,:,nodesToKeep]

        if self.allFunctionWords is not None:
            self.functionWords = [self.allFunctionWords[w] for w in nodesToKeep]
        
    def getGraph(self):
        
        return self.adjacencyMatrix
    
    def getFunctionWords(self):
        
        return self.functionWords, self.allFunctionWords
    
    def astype(self, dataType):
        # This changes the type for the selected author as well as the samples
        for key in self.selectedAuthor.keys():
            for secondKey in self.selectedAuthor[key].keys():
                self.selectedAuthor[key][secondKey] = changeDataType(
                                            self.selectedAuthor[key][secondKey],
                                            dataType)
        self.adjacencyMatrix = changeDataType(self.adjacencyMatrix, dataType)
        
        # And now, initialize to change the samples as well (and also save the 
        # data type)
        super().astype(dataType)
        
    
    def to(self, device):
        # If the dataType is 'torch'
        if 'torch' in repr(self.dataType):
            # Change the selected author ('test', 'train', 'valid', 'all';
            # 'WANs', 'wordFreq')
            for key in self.selectedAuthor.keys():
                for secondKey in self.selectedAuthor[key].keys():
                    self.selectedAuthor[key][secondKey] \
                                = self.selectedAuthor[key][secondKey].to(device)
            self.adjacencyMatrix.to(device)                 
            # And call the inherit method to initialize samples (and save to
            # device)
            super().to(device)
            
class MovieLens(_data):
    """
    MovieLens: Loads and handles handles the MovieLens-100k dataset

        The setting is that of regression on a specific node of the graph. That
        is, given a graph, and an incomplete graph signal on that graph, we want
        to estimate the value of the signal on a specific node.
        
        If, for instance, we have a movie-based graph, then the graph signal
        corresponds to the ratings that a given user gave to some of the movies.
        The objective is to estimate how that particular user would rate one
        of the other available movies. (Same holds by interchanging 'movie' with
        'user' in this paragraph)

    Initialization:

    Input:
        graphType('user' or 'movie'): which underlying graph to build; 'user'
            for user-based graph (each node is a user), and 'movie' for 
            movie-based (each node is a movie); this also determines the data,
            on a user-based graph, each data sample (each graph signal) 
            corresponds to a movie, and on the movie-based graph, each data
            sample corresponds to a user.
        labelID (list of int or 'all'): these are the specific nodes on which
            we will be looking to interpolate; this has effect in the building
            of the training, validation and test sets, since only data samples
            that have a value at that node can be used
        ratioTrain (float): ratio of the total samples to be part of the
            validation set
        ratioValid (float): ratio of the train samples to be part of the
            validation set
        dataDir (string): directory where to download the movie-lens dataset to/
            to check if it has already been downloaded
        keepIsolatedNodes (bool): If False, get rid of isolated nodes
        forceUndirected (bool): If True, create an undirected graph
        forceConnected (bool): If True, ensure that the resulting graph is
            connected
        kNN (int): sparsify this graph keeping kNN nearest neighbors
        maxNodes (int, default: None): consider only the maxNodes nodes with 
            largest number of ratings
        minRatings (int, default: 0): get rid of all columns and rows with
            less than minRatings ratings (for minRatings = 0, just keep all
            the matrix)
        interpolate (bool, default: False): if True, interpolates the matrix by
            means of a nearest-neighbor rule before creating the graph signals
            (i.e. all the graph signals will have complete ratings)
            >> Obs.: Just using these signals to interpolate the remaining 
                rating can be interpreted as a typical baseline.
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)
        
    The resulting dataset consists of triads (signal, target, labelID) where:
        - the signal contains the ratings given by some data sample to all nodes
          with a 0 for the rating corresponding to the labelID node (note that,
          if interpolate = False, then there will also be zeros in other nodes
          that have not been rated)
        - target is the value of the rating at the corresponding labelID node
        - labelID is the label of the node whose rating has been removed
        In other words, we want to use signal to estimate the value target at
        the node labelID.
    
    Methods:
        
    .loadData(filename, [dataDir]): loads the data from dataDir (if not
        provided, the internally stored one is used) and saves it as filename;
        if the data has already been processed and saved as 'filename', then
        it will be just loaded.
        
    .createGraph(): creates a graphType-based graph with the previously
        established options (undirected, isolated, connected, etc.); this graph 
        is always sparsified by means of a nearest-neighbor rule. The graph
        is created containing only data samples in the training set.
        
    .interpolateRatings(): uses a nearest-neighbor rule to interpolate the
        ratings in the graph signal; this means that all zero values that do not
        correspond to labelID are replaced by the average of ratings of the 
        closest neighbors with nonzero ratings.
    
    .getGraph(): fetches the adjacency matrix of the stored graph.
    
    .getIncompleteMatrix(): fetches the incomplete matrix as it was loaded
        from the data.
        
    .getMovieTitles(): fetches a dictionary, where each key is the movieID
        (starting from zero, so that it matches the index of the columns of
        the incomplete matrix; subtract 1 from the movieID of movieLens to 
        get this movieID) and each value is the title of the movie (in string
        format).
        
    .getLabelID(): the index of the node whose data will be regressed; this 
        might differ from the input labelID in that: its count starts from zero,
        its count might have been modified after getting rid of nodes in order
        to build a graph with the desired characteristics
        
    .evaluate(yHat, y): computes the RMSE between the estimated ratings yHat 
        and the actual ratings given in y.
        
    lossValue = .evaluate(yHat, y)
        Input:
            yHat (dtype.array): estimated target
            y (dtype.array): target representation

        Output:
            lossValue (float): regression loss chosen

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    """
    
    def __init__(self, graphType, labelID, ratioTrain, ratioValid, dataDir,
                 keepIsolatedNodes, forceUndirected, forceConnected, kNN,
                 maxNodes = None,
                 maxDataPoints = None,
                 minRatings = 0,
                 interpolate = False,
                 dataType = np.float64, device = 'cpu'):
        
        super().__init__()
        # This creates the attributes: dataType, device, nTrain, nTest, nValid,
        # and samples, and fills them all with None, and also creates the 
        # methods: getSamples, astype, and to.
        self.dataType = dataType
        self.device = device
        
        # Store attributes
        #   GraphType
        assert graphType == 'user' or graphType == 'movie'
        # This is because what are the graph signals depends on the graph we
        # want to use.
        self.graphType = graphType
        #   Label ID
        assert type(labelID) is list or labelID == 'all'
        # Label ID is the user ID or the movie ID following the MovieLens 
        # nomenclature. This determines how we build the labels in the
        # dataset. If it's all, then we want to estimate for all users/movies.
        #   Dataset partition
        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid
        #   Dataset directory
        self.dataDir = dataDir # Where the data is, or where it should be saved
        # to.
        #   Graph preferences
        self.keepIsolatedNodes = keepIsolatedNodes
        self.forceUndirected = forceUndirected
        self.forceConnected = forceConnected
        self.kNN = kNN
        #   Reduce the graph to have maxNodes
        self.maxNodes = maxNodes
        #   Discard samples with less than minRatings ratings
        self.minRatings = minRatings
        #   Interpolate nonexisting ratings (i.e. get rid of zeros and replace 
        #   them by the nearest neighbor rating)
        self.doInterpolate = interpolate
        #   Empty attributes for now
        self.incompleteMatrix = None
        self.movieTitles = {}
        self.adjacencyMatrix = None
        self.indexDataPoints = {}
        
        # Now, we should be ready to load the data and build the (incomplete) 
        # matrix
        self.loadData('movielens100kIncompleteMatrix.pkl')
        # This has loaded the incompleteMatrix and movieTitles attributes.
        
        # First check if we might need to get rid of columns and rows to get 
        # the minimum number of ratings requested
        if self.minRatings > 0:
            incompleteMatrix = self.incompleteMatrix
            # Get a one where there are ratings, and a 0 where there are not
            binaryIncompleteMatrix = (incompleteMatrix>0)\
                                                 .astype(incompleteMatrix.dtype)
            # Count the number of ratings in each row
            nRatingsPerRow = np.sum(binaryIncompleteMatrix, axis = 1)
            # Count the number of ratings in each column
            nRatingsPerCol = np.sum(binaryIncompleteMatrix, axis = 0)
            # Indices of rows and columns to keep
            indexRowsToKeep = np.nonzero(nRatingsPerRow > self.minRatings)[0]
            indexColsToKeep = np.nonzero(nRatingsPerCol > self.minRatings)[0]
            # Reduce the size of the matrix
            incompleteMatrix = \
                           incompleteMatrix[indexRowsToKeep][:, indexColsToKeep]
            # Store it
            self.incompleteMatrix = incompleteMatrix
            
            # Also, we need to consider that, if we have the movie graph, 
            # then we need to update the movie list as well (all the columns
            # we lost -the nodes we lost- are part of a movie list that
            # has a one-to-one correspondence)
            if self.graphType == 'movie':
                if len(self.movieTitles) > 0: # Non empty movieList
                    # Where to save the new movie list
                    movieTitles = {}
                    # Because nodes are now numbered sequentially, we need to
                    # do the same with the movieID to keep them matched (i.e.
                    # node n corresponds to movieList[n] title)
                    newMovieID = 0
                    for movieID in indexColsToKeep:
                        movieTitles[newMovieID] = self.movieTitles[movieID]
                        newMovieID = newMovieID + 1
                    # Update movieList
                    self.movieTitles = movieTitles
        else:
            # If there was no need to reduce the columns or rows
            indexRowsToKeep = np.arange(self.incompleteMatrix.shape[0])
            indexColsToKeep = np.arange(self.incompleteMatrix.shape[1])
        
        # To simplify code, we will work always with each row being a data
        # sample. The incompleteMatrix is User x Movies
        if graphType == 'user':
            # If we want to reduce the number of nodes (i.e. is not None), and
            # we want less nodes than the ones that actually there
            if maxNodes is not None and maxNodes<self.incompleteMatrix.shape[0]:
                # The number of columns in the matrix is the number of nodes,
                # therefore, each column is a node, and the number of nonzero
                # elements in each node is the number of ratings for each movie
                nRatings = np.sum((self.incompleteMatrix > zeroTolerance),
                                  axis = 1)
                # Order the nodes in decreasing order of number of ratings
                indexRowsToKeep = np.argsort(-nRatings)
                # Keep only the first nNodes
                indexRowsToKeep = indexRowsToKeep[0:maxNodes]
                # And reduce the size of the matrix
                self.incompleteMatrix = self.incompleteMatrix[indexRowsToKeep,:]
            # If the graph type is user-based, then the graph signals are the
            # movies, scored for every user. This means that each column of the
            # incompleteMatrix is a graph signal, but since we're working with
            # rows, we have to transpose it
            workingMatrix = self.incompleteMatrix.T # Movies x User
            # Which one correspond to the nodes
            indexNodesToKeep = indexRowsToKeep
            
            # Now, each row is a movie score for all users, so that it is a
            # graph signal in the user-based graph.
        else:
            if maxNodes is not None and maxNodes<self.incompleteMatrix.shape[1]:
                nRatings = np.sum((self.incompleteMatrix > zeroTolerance),
                                  axis = 0)
                indexColsToKeep = np.argsort(-nRatings)
                indexColsToKeep = indexColsToKeep[0:maxNodes]
                self.incompleteMatrix = self.incompleteMatrix[:,indexColsToKeep]
            workingMatrix = self.incompleteMatrix
            # In this case, each row is a user (how that user scored all movies)
            # and this is the kind of data samples we need for movie-based
            # graphs
            indexNodesToKeep = indexColsToKeep
        
        # Determine the number of nodes
        nNodes = workingMatrix.shape[1]
        
        assert len(indexNodesToKeep) == nNodes
        
        # And we need to map the original IDs to the new ones (note that
        # each column is a node now -each row is a graph signal- so we
        # care about matching the labels to the corresponding new ones)
        #   First check, that, unless we wanted all indices (so we don't
        #   care much about the ones we just dropped), we have them in the
        #   new indices (i.e. we didn't drop them)        
        if labelID != 'all':
            # For each of the introduced IDs, check:
            self.labelID = np.empty(0, dtype = np.int)
            for i in labelID:
                # Recall that labelID they start with 1, but indexNodesToKeep
                # starts with zero
                assert (i-1) in indexNodesToKeep
                newIndex = np.argwhere(indexNodesToKeep == (i-1))[0]
                self.labelID = np.concatenate((self.labelID, newIndex))
        else:
            self.labelID = np.arange(nNodes)
        
        # Up to this point, we just have an array of IDs of nodes we care about
        # This could be all, one or a few, but is a numpy.array
        
        # So, now we just select a number of rows (graph signals) at random
        # to make the train and valid and test set. But we need to keep
        # track of the ID (the node)
        # The total number of points is now the number of nonzero elements
        # of the matrix. The problem is that we cannot get a random number
        # of nonzero elements of the matrix, because we're risking selecting
        # all rows (graph signals), and thus not leaving anything for the
        # train and test set. In other words, the rows determine the graph
        # signals, and all the nonzero elements of each row will make up
        # for the points in each training set.
            
        # Next we reduce the size of the matrix to the ones that we are
        # interested in
        selectedMatrix = workingMatrix[:, self.labelID]
        
        # So far we've got the value of all graph signals only on the nodes
        # of interest (some of these might just be zero, if the nodes of
        # interest weren't rated by that given graph signal)
        
        # Get rid of those rows that have no ratings for the labels of
        # interest
        #   We sum all the rows: since all the ratings are positive, those
        #   rows that are zero is because they have no ratings
        nonzeroRows = np.sum(selectedMatrix, axis = 1)
        nonzeroRows = np.nonzero(nonzeroRows)[0]
        selectedMatrix = selectedMatrix[nonzeroRows,:]
        
        # Now, we move on to count the total number of graph signals that
        # we have (number of rows)
        nRows = selectedMatrix.shape[0]
        # Permute the indices at random
        randPerm = np.random.permutation(nRows)
        # This gives me a random way of going through all the rows. So we
        # will do that, going row by row, picking all the nonzero elements
        # in said row, until we reach the (closest possible) number to the
        # amount of training samples we want.
        # The point of this is that each row might have more than one
        # data point: i.e. some graph signal might have rated more than one
        # of the nodes of interest; therefore this would amount to having
        # more than one data point stemming from that graph signal -by 
        # zero-ing out each of the nodes separately-
        #   Total number of available samples (whether to take the 0 or the 
        #   1 element of the set is indistinct, they both have the same len)
        nDataPoints = len(np.nonzero(selectedMatrix)[0])
        #   Check if the total number of desired samples has been defined
        #   (a max number of data points could have been set if we want
        #   to randomly select a subset of all available datapoints, for
        #   running a faster training)
        if maxDataPoints is None:
            maxDataPoints = nDataPoints
        #   and if it was designed, if it is not greater than the total 
        #   number of data points available
        elif maxDataPoints > nDataPoints:
            maxDataPoints = nDataPoints
        self.maxDataPoints = maxDataPoints
        # Target number of train, valid and test samples
        nTrain = round(ratioTrain * maxDataPoints)
        nValid = round(ratioValid * nTrain)
        nTrain = nTrain - nValid
        nTest = maxDataPoints - nTrain - nValid
        
        # TODO: There has to be a way of accelerating this thing below
        
        # Training count
        nTrainSoFar = 0
        rowCounter = 0
        # Save variables
        trainSignals = np.empty([0, nNodes])
        trainLabels = np.empty(0)
        trainIDs = np.empty(0).astype(np.int)
        while nTrainSoFar < nTrain and rowCounter < nRows:
            # Get the corresponding selected row
            thisRow = selectedMatrix[randPerm[rowCounter], :]
            # Get the indices of the nonzero elements of interest (i.e
            # of all the nodes of interest, which ones have a nonzero
            # rating on this graph signal)
            thisNZcols = np.nonzero(thisRow)[0] # Nonzero Cols
            # And now we can match this to the corresponding columns in the
            # original matrix
            thisIDs = self.labelID[thisNZcols]
            thisNpoints = len(thisIDs)
            # Get the labels
            thisLabels = thisRow[thisNZcols]
            # Get the signals
            thisSignals = workingMatrix[nonzeroRows[randPerm[rowCounter]],:]
            # From this signal (taken from the original working matrix) we 
            # will obtain as many signals as nonzero ratings of the nodes of
            # interest. Therefore, we need to repeat it to that point
            thisSignals = np.tile(thisSignals, [thisNpoints, 1])
            #   thisNpoints x nNodes
            #   We need to zero-out those elements that will be part of
            #   the samples
            thisSignals[np.arange(thisNpoints), thisIDs] = 0
            # And now we should be able to concatenate
            trainSignals = np.concatenate((trainSignals, thisSignals),
                                          axis = 0)
            trainLabels = np.concatenate((trainLabels, thisLabels))
            trainIDs = np.concatenate((trainIDs, thisIDs))
            # Add how many new data points we have just got
            nTrainSoFar += thisNpoints
            # And increase the counter
            rowCounter += 1
        # We have finalized the training set. Now, we have to count how
        # many training samples we actually have
        self.nTrain = len(trainLabels)
        # We also want to know which rows we have selected so far
        indexTrainPoints = nonzeroRows[randPerm[0:rowCounter]]
        nRowsTrain = rowCounter
        
        # Now, repeat for validation set:
        nValidSoFar = 0
        rowCounter = nRowsTrain # Initialize where the other one left off
        # Save variables
        validSignals = np.empty([0, nNodes])
        validLabels = np.empty(0)
        validIDs = np.empty(0).astype(np.int)
        while nValidSoFar < nValid and rowCounter < nRows:
            # Get the corresponding selected row
            thisRow = selectedMatrix[randPerm[rowCounter], :]
            # Get the indices of the nonzero elements of interest (i.e
            # of all the nodes of interest, which ones have a nonzero
            # rating on this graph signal)
            thisNZcols = np.nonzero(thisRow)[0] # Nonzero Cols
            # And now we can match this to the corresponding columns in the
            # original matrix
            thisIDs = self.labelID[thisNZcols]
            thisNpoints = len(thisIDs)
            # Get the labels
            thisLabels = thisRow[thisNZcols]
            # Get the signals
            thisSignals = workingMatrix[nonzeroRows[randPerm[rowCounter]],:]
            # From this signal (taken from the original working matrix) we 
            # will obtain as many signals as nonzero ratings of the nodes of
            # interest. Therefore, we need to repeat it to that point
            thisSignals = np.tile(thisSignals, [thisNpoints, 1])
            #   thisNpoints x nNodes
            #   We need to zero-out those elements that will be part of
            #   the samples
            thisSignals[np.arange(thisNpoints), thisIDs] = 0
            # And now we should be able to concatenate
            validSignals = np.concatenate((validSignals, thisSignals),
                                          axis = 0)
            validLabels = np.concatenate((validLabels, thisLabels))
            validIDs = np.concatenate((validIDs, thisIDs))
            # Add how many new data points we have just got
            nValidSoFar += thisNpoints
            # And increase the counter
            rowCounter += 1
        # We have finalized the validation set. Now, we have to count how
        # many validation samples we actually have
        self.nValid = len(validLabels)
        # We also want to know which rows we have selected so far
        indexValidPoints = nonzeroRows[randPerm[nRowsTrain:rowCounter]]
        nRowsValid = rowCounter - nRowsTrain
        
        # And, finally the test set
        nTestSoFar = 0
        rowCounter = nRowsTrain + nRowsValid
        # Save variables
        testSignals = np.empty([0, nNodes])
        testLabels = np.empty(0)
        testIDs = np.empty(0).astype(np.int)
        while nTestSoFar < nTest and rowCounter < nRows:
            # Get the corresponding selected row
            thisRow = selectedMatrix[randPerm[rowCounter], :]
            # Get the indices of the nonzero elements of interest (i.e
            # of all the nodes of interest, which ones have a nonzero
            # rating on this graph signal)
            thisNZcols = np.nonzero(thisRow)[0] # Nonzero Cols
            # And now we can match this to the corresponding columns in the
            # original matrix
            thisIDs = self.labelID[thisNZcols]
            thisNpoints = len(thisIDs)
            # Get the labels
            thisLabels = thisRow[thisNZcols]
            # Get the signals
            thisSignals = workingMatrix[nonzeroRows[randPerm[rowCounter]],:]
            # From this signal (taken from the original working matrix) we 
            # will obtain as many signals as nonzero ratings of the nodes of
            # interest. Therefore, we need to repeat it to that point
            thisSignals = np.tile(thisSignals, [thisNpoints, 1])
            #   thisNpoints x nNodes
            #   We need to zero-out those elements that will be part of
            #   the samples
            thisSignals[np.arange(thisNpoints), thisIDs] = 0
            # And now we should be able to concatenate
            testSignals = np.concatenate((testSignals, thisSignals),
                                         axis = 0)
            testLabels = np.concatenate((testLabels, thisLabels))
            testIDs = np.concatenate((testIDs, thisIDs))
            # Add how many new data points we have just got
            nTestSoFar += thisNpoints
            # And increase the counter
            rowCounter += 1
        # We have finalized the validation set. Now, we have to count how
        # many validation samples we actually have
        self.nTest = len(testLabels)
        # We also want to know which rows we have selected so far
        indexTestPoints=nonzeroRows[randPerm[nRowsTrain+nRowsValid:rowCounter]]
        
        # And we also need all the data points (all the rows), so:
        indexDataPoints = np.concatenate((indexTrainPoints,
                                          indexValidPoints,
                                          indexTestPoints))
        
        # Now, this finalizes the data split, now, so we have all we need:
        # signals, labels, and IDs.
                
        # So far, either by selecting a node, or by selecting all nodes, we
        # have the variables we need: signals, labels, IDs and index points.
        self.samples['train']['signals'] = trainSignals
        self.samples['train']['targets'] = trainLabels
        self.samples['valid']['signals'] = validSignals
        self.samples['valid']['targets'] = validLabels
        self.samples['test']['signals'] = testSignals
        self.samples['test']['targets'] = testLabels
        self.targetIDs = {}
        self.targetIDs['train'] = trainIDs
        self.targetIDs['valid'] = validIDs
        self.targetIDs['test'] = testIDs
        # And update the index of the data points (which are the rows selected)
        self.indexDataPoints['all'] = indexDataPoints
        self.indexDataPoints['train'] = indexTrainPoints
        self.indexDataPoints['valid'] = indexValidPoints
        self.indexDataPoints['test'] = indexTestPoints
            
        # Now the data has been loaded, and the training/test partition has been
        # made, create the graph
        self.createGraph()
        # Observe that this graph also adjusts the signals to reflect any change
        # in the number of nodes
        
        # Finally, check if we want to interpolate the useless zeros
        if self.doInterpolate:
            self.interpolateRatings()
        
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def loadData(self, filename, *args):
        # Here we offer the option of including an additional dir, if not, use
        # the internally stored one.
        if len(args) == 1:
            dataDir = args[0]
        else:
            assert self.dataDir is not None
            dataDir = self.dataDir
        
        # Check if the dataDir exists, and if not, create it
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)
        # Create the filename to save/load
        datasetFilename = os.path.join(dataDir, filename)
        # Check if the file does exist, load it
        if os.path.isfile(datasetFilename):
            # If it exists, load it
            with open(datasetFilename, 'rb') as datasetFile:
                datasetDict = pickle.load(datasetFile)
                # And save the corresponding variable
                self.incompleteMatrix = datasetDict['incompleteMatrix']
                self.movieTitles = datasetDict['movieTitles']
        else: # If it doesn't exist, load it
            # There could be three options here: that we have the raw data 
            # already there, that we have the zip file and need to decompress it,
            # or that we do not have nothing and we need to download it.
            existsRawData = \
                      os.path.isfile(os.path.join(dataDir,'ml-100k','u.data')) \
                    and os.path.isfile(os.path.join(dataDir,'ml-100k','u.item'))
            # Actually, we're only interested in the ratings, but we're also
            # getting the movie list, just in case. Other information that we're
            # not considering at the moment includes: genres, user demographics
            existsZipFile = os.path.isfile(os.path.join(dataDir,'ml-100k.zip'))
            if not existsRawData and not existsZipFile: # We have to download it
                mlURL='http://files.grouplens.org/datasets/movielens/ml-100k.zip'
                urllib.request.urlretrieve(mlURL,
                                 filename = os.path.join(dataDir,'ml-100k.zip'))
                existsZipFile = True
            if not existsRawData and existsZipFile: # Unzip it
                zipObject = zipfile.ZipFile(os.path.join(dataDir,'ml-100k.zip'))
                zipObject.extractall(dataDir)
                zipObject.close()
            # Now that we have the data, we can get their filenames
            rawDataFilename = os.path.join(dataDir,'ml-100k','u.data')
            assert os.path.isfile(rawDataFilename)
            rawMovieListFilename = os.path.join(dataDir,'ml-100k','u.item')
            assert os.path.isfile(rawMovieListFilename)
            # And we can load it and store it.
            rawMatrix = np.empty([0, 0]) # Start with an empty matrix and then
            # we slowly add the number of users and movies, which we do not
            # assume to be known beforehand
            # Let's start with the data.
            # Open it.
            with open(rawDataFilename, 'r') as rawData:
                # The file consists of a succession of lines, each line
                # corresponds to a data sample
                for dataLine in rawData:
                    # For each line, we split it in the different fields
                    dataLineSplit = dataLine.rstrip('\n').split('\t')
                    # Keep the ones we care about here
                    userID = int(dataLineSplit[0])
                    movieID = int(dataLineSplit[1])
                    rating = int(dataLineSplit[2])
                    # Now we have to add this information to the matrix
                    # The matrix is of size Users x Movies (U x M)
                    #   We need to check whether we need to add more rows
                    #   or more columns
                    if userID > rawMatrix.shape[0]:
                        rowDiff = userID - rawMatrix.shape[0]
                        zeroPadRows = np.zeros([rowDiff, rawMatrix.shape[1]])
                        rawMatrix = np.concatenate((rawMatrix, zeroPadRows),
                                                   axis = 0)
                    if movieID > rawMatrix.shape[1]:
                        colDiff = movieID - rawMatrix.shape[1]
                        zeroPadCols = np.zeros([rawMatrix.shape[0], colDiff])
                        rawMatrix = np.concatenate((rawMatrix, zeroPadCols),
                                                   axis = 1)
                    # Now that we have assured appropriate dimensions
                    rawMatrix[userID - 1, movieID - 1] = rating
                    # Recall that the count of user and movie ID starts at 1
                    # for the movielens dataset, but we need to start indexing
                    # at 0 for Python
            # Now that we have created the matrix, we store it
            self.incompleteMatrix = rawMatrix
            # And we move to load the movie names
            
            with open(rawMovieListFilename, 'r', encoding = "ISO-8859-1") \
                    as rawMovieList:
                # Go line by line (each line corresponds to a movie)
                for movieLine in rawMovieList:
                    movieLineSplit = movieLine.rstrip('\n').split('|')
                    movieID = int(movieLineSplit[0]) - 1
                    # Look that, in this case, we're making the movies ID match
                    # the column indexing (so it starts at zero)
                    movieTitle = movieLineSplit[1]
                    self.movieTitles[movieID] = movieTitle
            # And now that we're done, we save this in a pickle file for
            # posterity
            with open(datasetFilename, 'wb') as datasetFile:
                pickle.dump(
                        {'incompleteMatrix': self.incompleteMatrix,
                         'movieTitles': self.movieTitles},
                        datasetFile
                        )
    
    def createGraph(self):
        # Here we can choose to create the movie or the user graph.
        # Let's start with the incomplete matrix, and get randomly some of the
        # elements from it to use as training data to build the graph.
        
        # Recall that the datapoints that I have already split following the
        # user/movie ID selection (or 'all' for all it matters) have to be
        # taken into account. So, check that this points have been determined
        assert 'all' in self.indexDataPoints.keys()
        assert 'train' in self.indexDataPoints.keys()
        assert 'valid' in self.indexDataPoints.keys()
        assert 'test' in self.indexDataPoints.keys()
        assert self.nTrain is not None \
                and self.nValid is not None \
                and self.nTest is not None            
        
        # To follow the paper by Huang et al., where the data is given by
        # Y in U x M, and goes into full detail on how to build the U x U
        # user-based graph, then, we will stick with this formulation
        if self.graphType == 'user':
            workingMatrix = self.incompleteMatrix # User x Movies
        else:
            workingMatrix = self.incompleteMatrix.T # Movies x User
        # Note that this is the opposite arrangement that we considered before
        # when loading the data into samples; back then, we considered samples
        # to be rows and the data to build the graph was therefore in columns;
        # in this case, it is the opposite, since we still want to use the data
        # located in the rows.
        
        # Now, the indices in self.indexDataPoints, essentially determine the
        # data samples (the graph signals) that we put in each set. Now, these
        # graph signals are now the columns, because the nodes are the rows.
        # So, these indexDataPoints are the columns in the new workingMatrix.
        
        # In essence, we need to add more points to complete the train set, but
        # to be sure that (i) these points are not the ones in the valid and
        # test sets, and (ii) that the training points are included already.
        
        # Now, out of all possible graph signals (number of columns in this
        # workingMatrix), we have selected some of those to be part of the
        # training set. But each  of these graph signals, have a different
        # number of traning points (because they have a different number of
        # nonzero elements). And we only care, when building the graph, on the
        # nonzero elements of the graph signals.
        
        # So, let's count the number of training points that we actually have
        # To do this, we count the number of nonzero elements in the samples
        # that we have selected
        trainSamples = self.indexDataPoints['train']
        nTrainPointsActual = len(np.nonzero(workingMatrix[:, trainSamples])[0])
        # And the total number of points that we have already partitioned into 
        # the different sets
        validSamples = self.indexDataPoints['valid']
        nValidPointsActual = len(np.nonzero(workingMatrix[:, validSamples])[0])
        testSamples = self.indexDataPoints['test']
        nTestPointsActual = len(np.nonzero(workingMatrix[:, testSamples])[0])
        # Total number of points already considered
        nPointsActual = nTrainPointsActual+nValidPointsActual+nTestPointsActual
        
        # The total number of data points in the entire dataset is
        indexDataPoints = np.nonzero(workingMatrix)
        # This is a tuple, where the first element is the place of nonzero
        # indices in the rows, and the second element is the place of nonzero
        # indices in the columns.
        nDataPoints = len(indexDataPoints[0]) # or [1], it doesn't matter
        # Note that every nonzero point belonging to labelID has already been
        # assigned to either one or the other dataset, so when we split
        # these datasets, we cannot consider these.
        
        # The total number of expected training points is
        nTrainPointsAll = int(round(self.ratioTrain * nDataPoints))
        #   Discard the (expected) number of validation points
        nTrainPointsAll = int(nTrainPointsAll\
                                        -round(self.ratioValid*nTrainPointsAll))
        
        # Now, we only need to add more points if the expected number of 
        # training points is greater than the ones we still have.
        
        # If we have more training points than what we originally intended, we
        # just use those (they will be part of the training samples regardless)
        # This could happen, for instance, if by chances, the graph signals
        # picked for training set are the more dense ones, giving a lot of
        # training points: nTrainPointsAll > nTrainPointsActual
        
        # Likewise, if we do not have any more points to take from (because all
        # the other graph signals have already been taken for validation and
        # test set), we can proceed to get the remaining needed points:
        # nPointsActual < nDataPoints
        
        if nTrainPointsAll > nTrainPointsActual and nPointsActual < nDataPoints:
            # So, now, the number of points that we still need to get are
            nTrainPointsRest = nTrainPointsAll - nTrainPointsActual
            # Next, we need to determine what is the pool of indices where we
            # can get the samples from (it cannot be samples that have already
            # been considered in any of the graph signals)
            nTotalCols = workingMatrix.shape[1] # Total number of columns
            # Note that self.indexDataPoints['all'] has all the columns that 
            # have already been selected. So the remaining columns are the ones
            # that are not there
            indexRemainingCols = [i for i in range(nTotalCols) \
                                        if i not in self.indexDataPoints['all']]
            indexRemainingCols = np.array(indexRemainingCols)
            # So the total number of points left is
            indexDataPointsRest=np.nonzero(workingMatrix[:,indexRemainingCols])
            nDataPointsRest = len(indexDataPointsRest[0])
            # Now, check that we have enough points to complete the total 
            # desired. If not, just use all of them
            if nDataPointsRest < nTrainPointsRest:
                nTrainPointsRest = nDataPointsRest
            
            # Now, we need to select at random from these points, those that 
            # will be part of the training set to build the graph.
            randPerm = np.random.permutation(nDataPointsRest)
            # Pick the needed number of subindices
            subIndexRandomRest = randPerm[0:nTrainPointsRest]
            # And select the points (both rows and columns)
            #   Remember that columns indexed by indexDataPointsRest, actually
            #   refer to the submatrix of remaining columns, so
            indexDataPointsRestCols=indexDataPointsRest[1][subIndexRandomRest]
            indexDataPointsRestCols=indexRemainingCols[indexDataPointsRestCols]
            indexDataPointsRestRows=indexDataPointsRest[0][subIndexRandomRest]
            indexTrainPointsRest = (indexDataPointsRestRows,
                                    indexDataPointsRestCols)
            # So, so far, we have all the needed training points: (i) those in
            # the original training set, and (ii) those in the remaining graph
            # signals to complete the number of desired training points.
            
            # Now, we need to merge these points with the ones already in the
            # training set of graph signals
            indexTrainPointsID = np.nonzero(
                                workingMatrix[:, self.indexDataPoints['train']])
            # And put them together with the ones we already had
            indexTrainPoints = (
                np.concatenate((indexTrainPointsRest[0],indexTrainPointsID[0])),
                np.concatenate((indexTrainPointsRest[1],
                           self.indexDataPoints['train'][indexTrainPointsID[1]]))
                                )
        else:
            # If we already had all the points we wanted, which are those that
            # were already in the training set, we need to get them, so it's
            # just a renaming
            indexTrainPoints = np.nonzero(
                                workingMatrix[:, self.indexDataPoints['train']])
            # But the columns in this indexTrainPoints, are actually the
            # columns of the smaller matrix evaluated only on 
            # self.indexDataPoints['train']. So we need to map it into the
            # full column numbers
            indexTrainPoints = (
                              indexTrainPoints[0],
                              self.indexDataPoints['train'][indexTrainPoints[1]]
                                )
            # And state that there are no new extra points
            nTrainPointsRest = 0
        
        # Record the actual number of training points that we are left with
        nTrainPoints = len(indexTrainPoints[0])
        assert nTrainPoints == nTrainPointsRest + nTrainPointsActual
        
        # And this is it! We got all the necessary training samples, including
        # those that we were already using.
        
        # Finally, set every other element not in the training set in the 
        # workingMatrix to zero
        workingMatrixZeroedTrain = workingMatrix.copy()      
        workingMatrixZeroedTrain[indexTrainPoints] = 0.
        workingMatrix = workingMatrix - workingMatrixZeroedTrain
        assert len(np.nonzero(workingMatrix)[0]) == nTrainPoints
        # To check that the total number of nonzero elements of the matrix are
        # the total number of training samples that we're supposed to have.
        
        # Now, we finally have the incompleteMatrix only with the corresponding
        # elements: a ratioTrain proportion of training samples that, for sure,
        # include the ones that we will use in the graph signals dataset and, 
        # for sure, exclude those that are in the validation and test sets.
       
        # Finally, on to compute the correlation matrix.
        # The mean required for the (u,v)th element of the correlation matrix is
        # the sum of the ratings for row u, but only in those columns where
        # there is also a rating for row v. So we care about the values in row
        # u, but we need to know which nonzero positions coincide between rows
        # u and v. In order to do this, we create a template that signals
        # the position of elements.
        binaryTemplate = (workingMatrix > 0).astype(workingMatrix.dtype)
        # Then, when we multiply the matrix with the actual ratings, with the
        # transpose of this template, we will be summing the values of one
        # matrix (in rows) but only for the places where there was an element
        # in the other row (now a column, because it is transposed). This gives
        # us the sum part of the mean.
        sumMatrix = workingMatrix.dot(binaryTemplate.T)
        # To count the number of elements that are shred by both rows u and v,
        # we simply multiply the binary template.
        countMatrix = binaryTemplate.dot(binaryTemplate.T)
        # Note that there might be elements with zero intersection, then we
        # need to set this to 1 so division by 0 doesn't create a NaN (the
        # end result will still be zero, since the sumMatrix will have a
        # zero in those same positions)
        countMatrix[countMatrix == 0] = 1
        # And now we can compute this (u,v) dependent mean
        avgMatrix = sumMatrix / countMatrix
        # Note that this matrix is not supposed to be symmetric due to the
        # use of only the sum of the item u over the set uv, instead of using
        # the sum over u and over v. More specifically, the definition is
        # mu_{uv} = 1/|S_{uv}| * \sum_{i \in S_{uv}} Y_{ui}
        # Since the sum is of elements u, when we compute mu_{vu} we will get
        # a different sum
        
        # Now, to compute the correlation, we need to compute the square sum
        # matrix \sum_{i \in S_{uv}} Y_{ui}^{2}
        sqSumMatrix = (workingMatrix ** 2).dot(binaryTemplate.T)
        # And compute the correlation matrix as
        # 1/|S_{uv}| \sum_{i \in S_{uv}} Y_{ui}^{2} - \mu_{uv}^{2}
        # where \mu_{uv} is the mean we computed before
        correlationMatrix = sqSumMatrix / countMatrix - avgMatrix ** 2
        
        # Finally, normalize the individual user variances and get rid of the
        # identity matrix
        #   Compute the square root of the diagonal elements
        sqrtDiagonal = np.sqrt(np.diag(correlationMatrix))
        #   Find the place where the nonzero elements are
        nonzeroSqrtDiagonalIndex = (sqrtDiagonal > zeroTolerance)\
                                                     .astype(sqrtDiagonal.dtype)
        #   Set the zero elements to 1
        sqrtDiagonal[sqrtDiagonal < zeroTolerance] = 1.
        #   Invert safely
        invSqrtDiagonal = 1/sqrtDiagonal
        #   Get rid of the fake 1 inversions
        invSqrtDiagonal = invSqrtDiagonal * nonzeroSqrtDiagonalIndex
        #   Make it a matrix again
        normalizationMatrix = np.diag(invSqrtDiagonal)
        #   And normalize
        normalizedMatrix = normalizationMatrix.dot(
                                correlationMatrix.dot(normalizationMatrix)) \
                            - np.eye(correlationMatrix.shape[0])
        #   There could be isolated nodes, which mean that have 0 in the 
        #   diagonal already, so when subtracting the identity they end up with
        #   -1 in the diagonal element.
        #   If this is the case, we just put back a one in those nodes. But the
        #   real problem, comes if the labelID is within those isolated nodes.
        #   If that's the case, then we just stop the processing, there's 
        #   nothing else to do.
        diagNormalizedMatrix = np.diag(np.diag(normalizedMatrix))
        isolatedNodes = np.nonzero(np.abs(diagNormalizedMatrix + 1) \
                                                                < zeroTolerance)
        normalizedMatrix[isolatedNodes] = 0.
        #   Get rid of the "quasi-zeros" that could have arrived through 
        #   division.
        normalizedMatrix[np.abs(normalizedMatrix) < zeroTolerance] = 0.
        
        # Finally, create the graph
        #   Number of nodes so far
        N = normalizedMatrix.shape[0]
        #   Add the necessary extra dimension (because it is 'fuseEdges' so it
        #   expects a set of matrices, instead of just one)
        normalizedMatrix = normalizedMatrix.reshape([1, N, N])
        #   Use 'fuseEdges' to handle several desirable properties that could
        #   be enforced on the graph
        nodesToKeep = [] # List of nodes to keep after some of them might have
        #   been removed to satisfy the constraints
        extraComponents= [] # List where we save the rest of the isolated 
        # components, if there where
        W = graph.createGraph('fuseEdges', N,
                              {'adjacencyMatrices': normalizedMatrix,
                               'aggregationType': 'sum',
                               'normalizationType': 'no',
                               'isolatedNodes': self.keepIsolatedNodes,
                               'forceUndirected': self.forceUndirected,
                               'forceConnected': self.forceConnected,
                               'nodeList': nodesToKeep,
                               'extraComponents': extraComponents})
        # So far, the matrix output is the adjacency matrix of the largest 
        # connected component, and nodesToKeep refer to those nodes.
        
        # At this point, it can happen that some (or all) of the selected nodes
        # are not in the graph. If none of the selected nodes is there, we
        # should stop (we have no useful problem anymore)
        
        IDnodesKept = 0 # How many of the selected ID nodes are we keeping
        for i in self.labelID:
            if i in nodesToKeep:
                IDnodesKept += 1
        
        assert IDnodesKept > 0
        
        #   Update samples and labelID, if necessary
        if len(nodesToKeep) < N:
            # Update the node IDs
            #   Get signals, IDs and labels
            trainSignals = self.samples['train']['signals']
            trainIDs = self.targetIDs['train']
            trainLabels = self.samples['train']['targets']
            validSignals = self.samples['valid']['signals']
            validIDs = self.targetIDs['valid']
            validLabels = self.samples['valid']['targets']
            testSignals = self.samples['test']['signals']
            testIDs = self.targetIDs['test']
            testLabels = self.samples['test']['targets']
            #   Update the ID
            #   Train set
            trainIDsToKeep = [] # which samples from the train set we need to 
                # keep (note that if some of the nodes that were labeled in the
                # trainIDs have been vanished, then we need to get rid of those
                # training samples)
            newTrainIDs = [] # Then, we need to match the old node numbering
                # (with all the nodes), to those of the new numbering
            for i in range(len(trainIDs)):
                # If the train ID of the sample is in nodes to keep
                if trainIDs[i] in nodesToKeep:
                    # We need to add it to the list of nodes to keep (what 
                    # position in the training samples are, because those that
                    # are not there also have to be discarded from the rating)
                    trainIDsToKeep.append(i)
                    # And we have to update the ID to the new one (considering
                    # that not all nodes have been kept)
                    newTrainIDs.append(nodesToKeep.index(trainIDs[i]))
            trainIDsToKeep = np.array(trainIDsToKeep) # Convert to numpy
            newTrainIDs = np.array(newTrainIDs) # Conver to numpy
            #   Valid Set
            validIDsToKeep = []
            newValidIDs = []
            for i in range(len(validIDs)):
                if validIDs[i] in nodesToKeep:
                    validIDsToKeep.append(i)
                    newValidIDs.append(nodesToKeep.index(validIDs[i]))
            validIDsToKeep = np.array(validIDsToKeep) # Convert to numpy
            newValidIDs = np.array(newValidIDs) # Conver to numpy
            #   Test Set
            testIDsToKeep = []
            newTestIDs = []
            for i in range(len(testIDs)):
                if testIDs[i] in nodesToKeep:
                    testIDsToKeep.append(i)
                    newTestIDs.append(nodesToKeep.index(testIDs[i]))
            testIDsToKeep = np.array(testIDsToKeep) # Convert to numpy
            newTestIDs = np.array(newTestIDs) # Conver to numpy
            
            # And, finally, we update the signals
            trainSignals = trainSignals[trainIDsToKeep][:, nodesToKeep]
            validSignals = validSignals[validIDsToKeep][:, nodesToKeep]
            testSignals = testSignals[testIDsToKeep][:, nodesToKeep]
            # and the IDs
            trainIDs = newTrainIDs
            validIDs = newValidIDs
            testIDs = newTestIDs
            # Also update the labels (some of the samples are gone)
            trainLabels = trainLabels[trainIDsToKeep]
            validLabels = validLabels[validIDsToKeep]
            testLabels = testLabels[testIDsToKeep]
            # and store them where they belong
            self.nTrain = trainSignals.shape[0]
            self.nValid = validSignals.shape[0]
            self.nTest = testSignals.shape[0]
            self.samples['train']['signals'] = trainSignals
            self.samples['train']['targets'] = trainLabels
            self.targetIDs['train'] = trainIDs
            self.samples['valid']['signals'] = validSignals
            self.samples['valid']['targets'] = validLabels
            self.targetIDs['valid'] = validIDs
            self.samples['test']['signals'] = testSignals
            self.samples['test']['targets'] = testLabels
            self.targetIDs['test'] = testIDs
            # If the graph type is 'movies', then any removed node has a
            # repercusion in the movie list, and therefore, we need to update
            # that as well
            if self.graphType == 'movie':
                if len(self.movieTitles) > 0: # Non empty movieList
                    # Where to save the new movie list
                    movieTitles = {}
                    # Because nodes are now numbered sequentially, we need to
                    # do the same with the movieID to keep them matched (i.e.
                    # node n corresponds to movieList[n] title)
                    newMovieID = 0
                    for movieID in nodesToKeep:
                        movieTitles[newMovieID] = self.movieTitles[movieID]
                        newMovieID = newMovieID + 1
                    # Update movieList
                    self.movieTitles = movieTitles
                
        #   And finally, sparsify it (nearest neighbors)
        self.adjacencyMatrix = graph.sparsifyGraph(W, 'NN', self.kNN)
        
    def interpolateRatings(self):
        # For the nonzero nodes, we will average the value of the closest
        # nonzero elements.
        
        # So we need to find the neighborhood, iteratively, until we find a
        # nodes in the neighborhood that have nonzero elements. And then
        # average those.
        
        # There are three sets of signals, so for each one of them
        for key in self.samples.keys():
            # Get the signals
            thisSignal = self.samples[key]['signals'] # B (x F) x N
            if len(thisSignal.shape) == 3: # If B x 1 x N
                assert thisSignal.shape[1] == 1
                thisSignal = thisSignal.squeeze(1) # B x N
            # Look for the elements in the signal that are zero
            zeroLocations = np.nonzero(np.abs(thisSignal) < zeroTolerance)
            # This is a tuple with two elements, each element is a 1-D np.array
            # with the rows and column indices of the zero elements, respectiv
            # The columns are the nodes, so we should iterate by nodes. The 
            # problem is that I do not want to go ahead finding the neighborhood
            # each time, and I do not want to go ahead element by element, I
            # want to go node by node, so that I have to do at most N searches
            
            # Not a good idea. Let's do it by neighborhood, since I can get
            # all neighbors at once
            # If there are zero locations that need to be interpolated
            K = 1
            while len(zeroLocations[0]) > 0:
                # Location of nodes with zero value
                zeroNodes = np.unique(zeroLocations[1])
                # If we want to make this faster, we only want the neighborhoods
                # of the nodes that don't have a value yet.
                # The problem is that the computeNeighborhood function only
                # works on the first N values, so we need to reorder the matrix
                # so that the first elements are the nodes we actually want
                # To do this, we need to add the rest of the nodes to the list
                # of zeroNodes and then reorder the matrix
                #   Full nodes
                fullNodes = [n for n in range(thisSignal.shape[1]) if n not in zeroNodes]
                fullNodes = np.array(fullNodes, dtype=np.int)
                #   Complete list of nodes (concatenate them)
                allNodes = np.concatenate((zeroNodes, fullNodes))
                #   Reorder the matrix
                A = self.adjacencyMatrix[allNodes, :][:, allNodes]
                # Get the neighborhood
                nbList = graph.computeNeighborhood(A, K, N = len(zeroNodes))
                # This is a list of lists. Each node has associated a list of
                # neighboring nodes.
                # But the index of this neighboring nodes is not correct, 
                # because it belongs to the allNodes ordering and not the 
                # original one.
                #   
                # Go for each node, and pick up the neighboring values
                # (It is more likely that we will have more samples than
                # nodes, so it should be faster to iterate through nodes)
                
                # For each element in the neighborhood list
                for i in range(len(nbList)):
                    # Get the actual node
                    thisNode = zeroNodes[i]
                    # Get the neighborhood (and map it to the corresponding
                    # nodes in the original ordering)
                    thisNB = [allNodes[n] for n in nbList[i]]
                    # Now, get the values at the neighborhood (which is now
                    # in the original ordering)
                    nbValues = thisSignal[:,thisNB]
                    #   This gives all the neighboring values of each batch
                    # Average the nonzero elements
                    #   Sum of the elements
                    sumSignal = np.sum(nbValues, axis = 1)
                    #   Count of nonzero elements
                    countNZ = np.count_nonzero(nbValues, axis = 1)
                    #   Get rid of the zero elements for division
                    countNZ[countNZ == 0] = 1.
                    #   Compute the average and round to an integer
                    meanSignal = np.round(sumSignal / countNZ)
                    # And now we need to place this newly computed mean 
                    # signal back in the nonzero elements
                    zeroBatches = zeroLocations[0][zeroLocations[1] == thisNode]
                    # Add it to the signal
                    thisSignal[zeroBatches, thisNode] = meanSignal[zeroBatches]
                # Now that we have finished all nodes for the K-hop neighbors
                # we need to update the zero elements
                zeroLocations = np.nonzero(np.abs(thisSignal) < zeroTolerance)
                # and add a new neighborhood
                K += 1
            
            # And put it back where it goes
            self.samples[key]['signals'] = thisSignal
        
    def getIncompleteMatrix(self):
        
        return self.incompleteMatrix
    
    def getGraph(self):
        
        return self.adjacencyMatrix
    
    def getMovieTitles(self):
        
        return self.movieTitles
    
    def getLabelID(self, *args):
        
        # So, here are the options
        # No arguments: return the list of self.labelID
        # One argument: it has to be samplesType and then return all labelIDs
        # for that sample type
        # Two arguments, can either be list or int, and return at random, like
        # the getSamples() method
        
        if len(args) == 0:
            returnID = self.labelID
        else:
            # The first argument has to be the sample type
            samplesType = args[0]
            # Check that is one of the possibilities
            assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
                    
            returnID = self.targetIDs[samplesType]
            
            if len(args) == 2:
                # If it is an int, just return that number of randomly chosen
                # IDs
                if type(args[1]) is int:
                    # Total number of samples
                    nSamples = returnID.shape
                    # Check that we are asked to return a number of samples that
                    # we actually have
                    assert args[1] <= nSamples
                    # Randomly choose args[1] indices
                    selectedIndices = np.random.choice(nSamples, size = args[1],
                                                       replace = False)
                    # Select the corresponding IDs
                    returnID = returnID[selectedIndices]
                
                else:
                    # This has to be a list () or an np.array which can serve
                    # as indexing functions
                    returnID = returnID[args[1]]
        
        return returnID
    
    def evaluate(self, yHat, y):
        # y and yHat should be of the same dimension, where dimension 0 is the
        # number of samples
        N = y.shape[0] # number of samples
        assert yHat.shape[0] == N
        # And now, get rid of any extra '1' dimension that might appear 
        # involuntarily from some vectorization issues.
        y = y.squeeze()
        yHat = yHat.squeeze()
        # Yet, if there was only one sample, then the sample dimension was
        # also get rid of during the squeeze, so we need to add it back
        if N == 1:
            y = y.unsqueeze(0)
            yHat = yHat.unsqueeze(0)
        
        # Now, we compute the RMS
        if 'torch' in repr(self.dataType):
            mse = torch.nn.functional.mse_loss(yHat, y)
            rmse = torch.sqrt(mse)
        else:
            mse = np.mean((yHat - y) ** 2) 
            rmse = np.sqrt(mse)
            
        return rmse
    
    def astype(self, dataType):
        # This changes the type for the incomplete and adjacency matrix.
        self.incompleteMatrix = changeDataType(self.incompleteMatrix, dataType)
        self.adjacencyMatrix = changeDataType(self.adjacencyMatrix, dataType)
        
        # And now, initialize to change the samples as well (and also save the 
        # data type)
        super().astype(dataType)
        
    
    def to(self, device):
        # If the dataType is 'torch'
        if 'torch' in repr(self.dataType):
            # Change the stored attributes that are not handled by the inherited
            # method to().
            self.incompleteMatrix.to(device)
            self.adjacencyMatrix.to(device)
            # And call the inherit method to initialize samples (and save to
            # device)
            super().to(device)
            
class Flocking(_data):
    """
    Flocking: Creates synthetic trajectories for the problem of coordinating
        a team of robots to fly together while avoiding collisions. See the
        following  paper for details
        
        E. Tolstaya, F. Gama, J. Paulos, G. Pappas, V. Kumar, and A. Ribeiro, 
        "Learning Decentralized Controllers for Robot Swarms with Graph Neural
        Networks," in Conf. Robot Learning 2019. Osaka, Japan: Int. Found.
        Robotics Res., 30 Oct.-1 Nov. 2019.
    
    Initialization:
        
    Input:
        nAgents (int): Number of agents
        commRadius (float): communication radius (in meters)
        repelDist (float): minimum target separation of agents (in meters)
        nTrain (int): number of training trajectories
        nValid (int): number of validation trajectories
        nTest (int): number of testing trajectories
        duration (float): duration of each trajectory (in seconds)
        samplingTime (float): time between consecutive time instants (in sec)
        initGeometry ('circular', 'rectangular'): initial positioning geometry
            (default: 'circular')
        initVelValue (float): maximum initial velocity (in meters/seconds,
            default: 3.)
        initMinDist (float): minimum initial distance between agents (in
            meters, default: 0.1)
        accelMax (float): maximum possible acceleration (in meters/seconds^2,
            default: 10.)
        normalizeGraph (bool): if True normalizes the communication graph
            adjacency matrix by the maximum eigenvalue (default: True)
        doPrint (bool): If True prints messages (default: True)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved (default: 'cpu')
            
    Methods:
        
    signals, targets = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x 6 x numberNodes
            targets (dtype.array): numberSamples x 2 x numberNodes
            'signals' are the state variables as described in the corresponding
            paper; 'targets' is the 2-D acceleration for each node
            
    cost = .evaluate(vel = None, accel = None, initVel = None,
                     samplingTime = None)
        Input:
            vel (array): velocities; nSamples x tSamples x 2 x nAgents
            accel (array): accelerations; nSamples x tSamples x 2 x nAgents
            initVel (array): initial velocities; nSamples x 2 x nAgents
            samplingTime (float): sampling time
            >> Obs.: Either vel or (accel and initVel) have to be specified
            for the cost to be computed, if all of them are specified, only
            vel is used
        Output:
            cost (float): flocking cost as specified in eq. (13)

    .astype(dataType): change the type of the data matrix arrays.
        Input:
            dataType (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 
                'cpu', 'cuda:0', etc.)

    state = .computeStates(pos, vel, graphMatrix, ['doPrint'])
        Input:
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            vel (array): velocities; nSamples x tSamples x 2 x nAgents
            graphMatrix (array): matrix description of communication graph;
                nSamples x tSamples x nAgents x nAgents
            'doPrint' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
        Output:
            state (array): states; nSamples x tSamples x 6 x nAgents
    
    graphMatrix = .computeCommunicationGraph(pos, commRadius, normalizeGraph,
                    ['kernelType' = 'gaussian', 'weighted' = False, 'doPrint'])
        Input:
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            commRadius (float): communication radius (in meters)
            normalizeGraph (bool): if True normalize adjacency matrix by 
                largest eigenvalue
            'kernelType' ('gaussian'): kernel to apply to the distance in order
                to compute the weights of the adjacency matrix, default is
                the 'gaussian' kernel; other kernels have to be coded, and also
                the parameters of the kernel have to be included as well, in
                the case of the gaussian kernel, 'kernelScale' determines the
                scale (default: 1.)
            'weighted' (bool): if True the graph is weighted according to the
                kernel type; if False, it's just a binary adjacency matrix
            'doPrint' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
        Output:
            graphMatrix (array): adjacency matrix of the communication graph;
                nSamples x tSamples x nAgents x nAgents
    
    thisData = .getData(name, samplesType[, optionalArguments])
        Input:
            name (string): variable name to get (for example, 'pos', 'vel', 
                etc.)
            samplesType ('train', 'test' or 'valid')
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            thisData (array): specific type of data requested
    
    pos, vel[, accel, state, graph] = computeTrajectory(initPos, initVel,
                                            duration[, 'archit', 'accel',
                                            'doPrint'])
        Input:
            initPos (array): initial positions; nSamples x 2 x nAgents
            initVel (array): initial velocities; nSamples x 2 x nAgents
            duration (float): duration of trajectory (in seconds)
            Optional arguments: (either 'accel' or 'archit' have to be there)
            'archit' (nn.Module): torch architecture that computes the output
                from the states
            'accel' (array): accelerations; nSamples x tSamples x 2 x nAgents
            'doPrint' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
        Output:
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            vel (array): velocities; nSamples x tSamples x 2 x nAgents
            Optional outputs (only if 'archit' was used)
            accel (array): accelerations; nSamples x tSamples x 2 x nAgents
            state (array): state; nSamples x tSamples x 6 x nAgents
            graph (array): adjacency matrix of communication graph;
                nSamples x tSamples x nAgents x nAgents
            
    uDiff, uDistSq = .computeDifferences (u):
        Input:
            u (array): nSamples (x tSamples) x 2 x nAgents
        Output:
            uDiff (array): pairwise differences between the agent entries of u;
                nSamples (x tSamples) x 2 x nAgents x nAgents
            uDistSq (array): squared distances between agent entries of u;
                nSamples (x tSamples) x nAgents x nAgents
    
    pos, vel, accel = .computeOptimalTrajectory(initPos, initVel, duration, 
                                                samplingTime, repelDist,
                                                accelMax = 100.)
        Input:
            initPos (array): initial positions; nSamples x 2 x nAgents
            initVel (array): initial velocities; nSamples x 2 x nAgents
            duration (float): duration of trajectory (in seconds)
            samplingTime (float): time elapsed between consecutive time 
                instants (in seconds)
            repelDist (float): minimum desired distance between agents (in m)
            accelMax (float, default = 100.): maximum possible acceleration
        Output:
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            vel (array): velocities; nSamples x tSamples x 2 x nAgents
            accel (array): accelerations; nSamples x tSamples x 2 x nAgents
            
    initPos, initVel = .computeInitialPositions(nAgents, nSamples, commRadius,
                                                minDist = 0.1,
                                                geometry = 'rectangular',
                                                xMaxInitVel = 3.,
                                                yMaxInitVel = 3.)
        Input:
            nAgents (int): number of agents
            nSamples (int): number of sample trajectories
            commRadius (float): communication radius (in meters)
            minDist (float): minimum initial distance between agents (in m)
            geometry ('rectangular', 'circular'): initial geometry
            xMaxInitVel (float): maximum velocity in the x-axis
            yMaxInitVel (float): maximum velocity in the y-axis
        Output:
            initPos (array): initial positions; nSamples x 2 x nAgents
            initVel (array): initial velocities; nSamples x 2 x nAgents
    
    .saveVideo(saveDir, pos, [, optionalArguments], commGraph = None,
               [optionalKeyArguments])
        Input:
            saveDir (os.path, string): directory where to save the trajectory
                videos
            pos (array): positions; nSamples x tSamples x 2 x nAgents
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
            commGraph (array): adjacency matrix of communication graph;
                nSamples x tSamples x nAgents x nAgents
                if not None, then this array is used to produce snapshots of
                the video that include the communication graph at that time
                instant
            'doPrint' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
            'videoSpeed' (float): how faster or slower the video is reproduced
                (default: 1.)
            'showVideoSpeed' (bool): if True shows the legend with the video
                speed in the video; by default it will show it whenever the
                video speed is different from 1.
            'vel' (array): velocities; nSamples x tSamples x 2 x nAgents
            'showCost' (bool): if True and velocities are set, the snapshots
                will show the instantaneous cost (default: True)
            'showArrows' (bool): if True and velocities are set, the snapshots
                will show the arrows of the velocities (default: True)
            
            
    """
    
    def __init__(self, nAgents, commRadius, repelDist,
                 nTrain, nValid, nTest,
                 duration, samplingTime,
                 initGeometry = 'circular',initVelValue = 3.,initMinDist = 0.1,
                 accelMax = 10.,
                 normalizeGraph = True, doPrint = True,
                 dataType = np.float64, device = 'cpu'):
        
        # Initialize parent class
        super().__init__()
        # Save the relevant input information
        #   Number of nodes
        self.nAgents = nAgents
        self.commRadius = commRadius
        self.repelDist = repelDist
        #   Number of samples
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        nSamples = nTrain + nValid + nTest
        #   Geometry
        self.mapWidth = None
        self.mapHeight = None
        #   Agents
        self.initGeometry = initGeometry
        self.initVelValue = initVelValue
        self.initMinDist = initMinDist
        self.accelMax = accelMax
        #   Duration of the trajectory
        self.duration = float(duration)
        self.samplingTime = samplingTime
        #   Data
        self.normalizeGraph = normalizeGraph
        self.dataType = dataType
        self.device = device
        #   Options
        self.doPrint = doPrint
        
        #   Places to store the data
        self.initPos = None
        self.initVel = None
        self.pos = None
        self.vel = None
        self.accel = None
        self.commGraph = None
        self.state = None
        
        if self.doPrint:
            print("\tComputing initial conditions...", end = ' ', flush = True)
        
        # Compute the initial positions
        initPosAll, initVelAll = self.computeInitialPositions(
                                          self.nAgents, nSamples, self.commRadius,
                                          minDist = self.initMinDist,
                                          geometry = self.initGeometry,
                                          xMaxInitVel = self.initVelValue,
                                          yMaxInitVel = self.initVelValue
                                                              )
        #   Once we have all positions and velocities, we will need to split 
        #   them in the corresponding datasets (train, valid and test)
        self.initPos = {}
        self.initVel = {}
        
        if self.doPrint:
            print("OK", flush = True)
            # Erase the label first, then print it
            print("\tComputing the optimal trajectories...",
                  end=' ', flush=True)
        
        # Compute the optimal trajectory
        posAll, velAll, accelAll = self.computeOptimalTrajectory(
                                        initPosAll, initVelAll, self.duration,
                                        self.samplingTime, self.repelDist,
                                        accelMax = self.accelMax)
        
        self.pos = {}
        self.vel = {}
        self.accel = {}
        
        if self.doPrint:
            print("OK", flush = True)
            # Erase the label first, then print it
            print("\tComputing the communication graphs...",
                  end=' ', flush=True)
        
        # Compute communication graph
        commGraphAll = self.computeCommunicationGraph(posAll, self.commRadius,
                                                      self.normalizeGraph)
        
        self.commGraph = {}
        
        if self.doPrint:
            print("OK", flush = True)
            # Erase the label first, then print it
            print("\tComputing the agent states...", end = ' ', flush = True)
        
        # Compute the states
        stateAll = self.computeStates(posAll, velAll, commGraphAll)
        
        self.state = {}
        
        if self.doPrint:
            # Erase the label
            print("OK", flush = True)
        
        # Separate the states into training, validation and testing samples
        # and save them
        #   Training set
        self.samples['train']['signals'] = stateAll[0:self.nTrain].copy()
        self.samples['train']['targets'] = accelAll[0:self.nTrain].copy()
        self.initPos['train'] = initPosAll[0:self.nTrain]
        self.initVel['train'] = initVelAll[0:self.nTrain]
        self.pos['train'] = posAll[0:self.nTrain]
        self.vel['train'] = velAll[0:self.nTrain]
        self.accel['train'] = accelAll[0:self.nTrain]
        self.commGraph['train'] = commGraphAll[0:self.nTrain]
        self.state['train'] = stateAll[0:self.nTrain]
        #   Validation set
        startSample = self.nTrain
        endSample = self.nTrain + self.nValid
        self.samples['valid']['signals']=stateAll[startSample:endSample].copy()
        self.samples['valid']['targets']=accelAll[startSample:endSample].copy()
        self.initPos['valid'] = initPosAll[startSample:endSample]
        self.initVel['valid'] = initVelAll[startSample:endSample]
        self.pos['valid'] = posAll[startSample:endSample]
        self.vel['valid'] = velAll[startSample:endSample]
        self.accel['valid'] = accelAll[startSample:endSample]
        self.commGraph['valid'] = commGraphAll[startSample:endSample]
        self.state['valid'] = stateAll[startSample:endSample]
        #   Testing set
        startSample = self.nTrain + self.nValid
        endSample = self.nTrain + self.nValid + self.nTest
        self.samples['test']['signals']=stateAll[startSample:endSample].copy()
        self.samples['test']['targets']=accelAll[startSample:endSample].copy()
        self.initPos['test'] = initPosAll[startSample:endSample]
        self.initVel['test'] = initVelAll[startSample:endSample]
        self.pos['test'] = posAll[startSample:endSample]
        self.vel['test'] = velAll[startSample:endSample]
        self.accel['test'] = accelAll[startSample:endSample]
        self.commGraph['test'] = commGraphAll[startSample:endSample]
        self.state['test'] = stateAll[startSample:endSample]
        
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def astype(self, dataType):
        
        # Change all other signals to the correct place
        datasetType = ['train', 'valid', 'test']
        for key in datasetType:
            self.initPos[key] = changeDataType(self.initPos[key], dataType)
            self.initVel[key] = changeDataType(self.initVel[key], dataType)
            self.pos[key] = changeDataType(self.pos[key], dataType)
            self.vel[key] = changeDataType(self.vel[key], dataType)
            self.accel[key] = changeDataType(self.accel[key], dataType)
            self.commGraph[key] = changeDataType(self.commGraph[key], dataType)
            self.state[key] = changeDataType(self.state[key], dataType)
        
        # And call the parent
        super().astype(dataType)
        
    def to(self, device):
        
        # Check the data is actually torch
        if 'torch' in repr(self.dataType):
            datasetType = ['train', 'valid', 'test']
            # Move the data
            for key in datasetType:
                self.initPos[key].to(device)
                self.initVel[key].to(device)
                self.pos[key].to(device)
                self.vel[key].to(device)
                self.accel[key].to(device)
                self.commGraph[key].to(device)
                self.state[key].to(device)
            
            super().to(device)
            
    def expandDims(self):
        # Just avoid the 'expandDims' method in the parent class
        pass
        
    def computeStates(self, pos, vel, graphMatrix, **kwargs):
        
        # We get the following inputs.
        # positions: nSamples x tSamples x 2 x nAgents
        # velocities: nSamples x tSamples x 2 x nAgents
        # graphMatrix: nSaples x tSamples x nAgents x nAgents
        
        # And we want to build the state, which is a vector of dimension 6 on 
        # each node, that is, the output shape is
        #   nSamples x tSamples x 6 x nAgents
        
        # The print for this one can be settled independently, if not, use the
        # default of the data object
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint
        
        # Check correct dimensions
        assert len(pos.shape) == len(vel.shape) == len(graphMatrix.shape) == 4
        nSamples = pos.shape[0]
        tSamples = pos.shape[1]
        assert pos.shape[2] == 2
        nAgents = pos.shape[3]
        assert vel.shape[0] == graphMatrix.shape[0] == nSamples
        assert vel.shape[1] == graphMatrix.shape[1] == tSamples
        assert vel.shape[2] == 2
        assert vel.shape[3] == graphMatrix.shape[2] == graphMatrix.shape[3] \
                == nAgents
                
        # If we have a lot of batches and a particularly long sequence, this
        # is bound to fail, memory-wise, so let's do it time instant by time
        # instant if we have a large number of time instants, and split the
        # batches
        maxTimeSamples = 200 # Set the maximum number of t.Samples before
            # which to start doing this time by time.
        maxBatchSize = 100 # Maximum number of samples to process at a given
            # time
        
        # Compute the number of samples, and split the indices accordingly
        if nSamples < maxBatchSize:
            nBatches = 1
            batchSize = [nSamples]
        elif nSamples % maxBatchSize != 0:
            # If we know it's not divisible, then we do floor division and
            # add one more batch
            nBatches = nSamples // maxBatchSize + 1
            batchSize = [maxBatchSize] * nBatches
            # But the last batch is actually smaller, so just add the 
            # remaining ones
            batchSize[-1] = nSamples - sum(batchSize[0:-1])
        # If they fit evenly, then just do so.
        else:
            nBatches = int(nSamples/maxBatchSize)
            batchSize = [maxBatchSize] * nBatches
        # batchIndex is used to determine the first and last element of each
        # batch. We need to add the 0 because it's the first index.
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        # Create the output state variable
        state = np.zeros((nSamples, tSamples, 6, nAgents))
        
        for b in range(nBatches):
            
            # Pick the batch elements
            posBatch = pos[batchIndex[b]:batchIndex[b+1]]
            velBatch = vel[batchIndex[b]:batchIndex[b+1]]
            graphMatrixBatch = graphMatrix[batchIndex[b]:batchIndex[b+1]]
        
            if tSamples > maxTimeSamples:
                
                # For each time instant
                for t in range(tSamples):
                    
                    # Now, we need to compute the differences, in velocities and in 
                    # positions, for each agent, for each time instant
                    posDiff, posDistSq = \
                                     self.computeDifferences(posBatch[:,t,:,:])
                    #   posDiff: batchSize[b] x 2 x nAgents x nAgents
                    #   posDistSq: batchSize[b] x nAgents x nAgents
                    velDiff, _ = self.computeDifferences(velBatch[:,t,:,:])
                    #   velDiff: batchSize[b] x 2 x nAgents x nAgents
                    
                    # Next, we need to get ride of all those places where there are
                    # no neighborhoods. That is given by the nonzero elements of the 
                    # graph matrix.
                    graphMatrixTime = (np.abs(graphMatrixBatch[:,t,:,:])\
                                                               >zeroTolerance)\
                                                             .astype(pos.dtype)
                    #   graphMatrix: batchSize[b] x nAgents x nAgents
                    # We also need to invert the squares of the distances
                    posDistSqInv = invertTensorEW(posDistSq)
                    #   posDistSqInv: batchSize[b] x nAgents x nAgents
                    
                    # Now we add the extra dimensions so that all the 
                    # multiplications are adequate
                    graphMatrixTime = np.expand_dims(graphMatrixTime, 1)
                    #   graphMatrix: batchSize[b] x 1 x nAgents x nAgents
                    
                    # Then, we can get rid of non-neighbors
                    posDiff = posDiff * graphMatrixTime
                    posDistSqInv = np.expand_dims(posDistSqInv,1)\
                                                              * graphMatrixTime
                    velDiff = velDiff * graphMatrixTime
                    
                    # Finally, we can compute the states
                    stateVel = np.sum(velDiff, axis = 3)
                    #   stateVel: batchSize[b] x 2 x nAgents
                    statePosFourth = np.sum(posDiff * (posDistSqInv ** 2),
                                            axis = 3)
                    #   statePosFourth: batchSize[b] x 2 x nAgents
                    statePosSq = np.sum(posDiff * posDistSqInv, axis = 3)
                    #   statePosSq: batchSize[b] x 2 x nAgents
                    
                    # Concatentate the states and return the result
                    state[batchIndex[b]:batchIndex[b+1],t,:,:] = \
                                                np.concatenate((stateVel,
                                                                statePosFourth,
                                                                statePosSq),
                                                               axis = 1)
                    #   batchSize[b] x 6 x nAgents
                    
                    if doPrint:
                        # Sample percentage count
                        percentageCount = int(100*(t+1+b*tSamples)\
                                                          /(nBatches*tSamples))
                        
                        if t == 0 and b == 0:
                            # It's the first one, so just print it
                            print("%3d%%" % percentageCount,
                                  end = '', flush = True)
                        else:
                            # Erase the previous characters
                            print('\b \b' * 4 + "%3d%%" % percentageCount,
                                  end = '', flush = True)
                
            else:
                
                # Now, we need to compute the differences, in velocities and in 
                # positions, for each agent, for each time instante
                posDiff, posDistSq = self.computeDifferences(posBatch)
                #   posDiff: batchSize[b] x tSamples x 2 x nAgents x nAgents
                #   posDistSq: batchSize[b] x tSamples x nAgents x nAgents
                velDiff, _ = self.computeDifferences(velBatch)
                #   velDiff: batchSize[b] x tSamples x 2 x nAgents x nAgents
                
                # Next, we need to get ride of all those places where there are
                # no neighborhoods. That is given by the nonzero elements of the 
                # graph matrix.
                graphMatrixBatch = (np.abs(graphMatrixBatch) > zeroTolerance)\
                                                             .astype(pos.dtype)
                #   graphMatrix: batchSize[b] x tSamples x nAgents x nAgents
                # We also need to invert the squares of the distances
                posDistSqInv = invertTensorEW(posDistSq)
                #   posDistSqInv: batchSize[b] x tSamples x nAgents x nAgents
                
                # Now we add the extra dimensions so that all the multiplications
                # are adequate
                graphMatrixBatch = np.expand_dims(graphMatrixBatch, 2)
                #   graphMatrix:batchSize[b] x tSamples x 1 x nAgents x nAgents
                
                # Then, we can get rid of non-neighbors
                posDiff = posDiff * graphMatrixBatch
                posDistSqInv = np.expand_dims(posDistSqInv, 2)\
                                                             * graphMatrixBatch
                velDiff = velDiff * graphMatrixBatch
                
                # Finally, we can compute the states
                stateVel = np.sum(velDiff, axis = 4)
                #   stateVel: batchSize[b] x tSamples x 2 x nAgents
                statePosFourth = np.sum(posDiff * (posDistSqInv ** 2), axis = 4)
                #   statePosFourth: batchSize[b] x tSamples x 2 x nAgents
                statePosSq = np.sum(posDiff * posDistSqInv, axis = 4)
                #   statePosSq: batchSize[b] x tSamples x 2 x nAgents
                
                # Concatentate the states and return the result
                state[batchIndex[b]:batchIndex[b+1]] = \
                                                np.concatenate((stateVel,
                                                                statePosFourth,
                                                                statePosSq),
                                                               axis = 2)
                #   state: batchSize[b] x tSamples x 6 x nAgents
                                                
                if doPrint:
                    # Sample percentage count
                    percentageCount = int(100*(b+1)/nBatches)
                    
                    if b == 0:
                        # It's the first one, so just print it
                        print("%3d%%" % percentageCount,
                              end = '', flush = True)
                    else:
                        # Erase the previous characters
                        print('\b \b' * 4 + "%3d%%" % percentageCount,
                              end = '', flush = True)
                        
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
        
        return state
        
    def computeCommunicationGraph(self, pos, commRadius, normalizeGraph,
                                  **kwargs):
        
        # Take in the position and the communication radius, and return the
        # trajectory of communication graphs
        # Input will be of shape
        #   nSamples x tSamples x 2 x nAgents
        # Output will be of shape
        #   nSamples x tSamples x nAgents x nAgents
        
        assert commRadius > 0
        assert len(pos.shape) == 4
        nSamples = pos.shape[0]
        tSamples = pos.shape[1]
        assert pos.shape[2] == 2
        nAgents = pos.shape[3]
        
        # Graph type options
        #   Kernel type (only Gaussian implemented so far)
        if 'kernelType' in kwargs.keys():
            kernelType = kwargs['kernelType']
        else:
            kernelType = 'gaussian'
        #   Decide if the graph is weighted or not
        if 'weighted' in kwargs.keys():
            weighted = kwargs['weighted']
        else:
            weighted = False
        
        # If it is a Gaussian kernel, we need to determine the scale
        if kernelType == 'gaussian':
            if 'kernelScale' in kwargs.keys():
                kernelScale = kwargs['kernelScale']
            else:
                kernelScale = 1.
        
        # The print for this one can be settled independently, if not, use the
        # default of the data object
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint
                
        # If we have a lot of batches and a particularly long sequence, this
        # is bound to fail, memory-wise, so let's do it time instant by time
        # instant if we have a large number of time instants, and split the
        # batches
        maxTimeSamples = 200 # Set the maximum number of t.Samples before
            # which to start doing this time by time.
        maxBatchSize = 100 # Maximum number of samples to process at a given
            # time
        
        # Compute the number of samples, and split the indices accordingly
        if nSamples < maxBatchSize:
            nBatches = 1
            batchSize = [nSamples]
        elif nSamples % maxBatchSize != 0:
            # If we know it's not divisible, then we do floor division and
            # add one more batch
            nBatches = nSamples // maxBatchSize + 1
            batchSize = [maxBatchSize] * nBatches
            # But the last batch is actually smaller, so just add the 
            # remaining ones
            batchSize[-1] = nSamples - sum(batchSize[0:-1])
        # If they fit evenly, then just do so.
        else:
            nBatches = int(nSamples/maxBatchSize)
            batchSize = [maxBatchSize] * nBatches
        # batchIndex is used to determine the first and last element of each
        # batch. We need to add the 0 because it's the first index.
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        # Create the output state variable
        graphMatrix = np.zeros((nSamples, tSamples, nAgents, nAgents))
        
        for b in range(nBatches):
            
            # Pick the batch elements
            posBatch = pos[batchIndex[b]:batchIndex[b+1]]
                
            if tSamples > maxTimeSamples:
                # If the trajectories are longer than 200 points, then do it 
                # time by time.
                
                # For each time instant
                for t in range(tSamples):
                    
                    # Let's start by computing the distance squared
                    _, distSq = self.computeDifferences(posBatch[:,t,:,:])
                    # Apply the Kernel
                    if kernelType == 'gaussian':
                        graphMatrixTime = np.exp(-kernelScale * distSq)
                    else:
                        graphMatrixTime = distSq
                    # Now let's place zeros in all places whose distance is greater
                    # than the radius
                    graphMatrixTime[distSq > (commRadius ** 2)] = 0.
                    # Set the diagonal elements to zero
                    graphMatrixTime[:,\
                                    np.arange(0,nAgents),np.arange(0,nAgents)]\
                                                                           = 0.
                    # If it is unweighted, force all nonzero values to be 1
                    if not weighted:
                        graphMatrixTime = (graphMatrixTime > zeroTolerance)\
                                                          .astype(distSq.dtype)
                                                              
                    if normalizeGraph:
                        isSymmetric = np.allclose(graphMatrixTime,
                                                  np.transpose(graphMatrixTime,
                                                               axes = [0,2,1]))
                        # Tries to make the computation faster, only the 
                        # eigenvalues (while there is a cost involved in 
                        # computing whether the matrix is symmetric, 
                        # experiments found that it is still faster to use the
                        # symmetric algorithm for the eigenvalues)
                        if isSymmetric:
                            W = np.linalg.eigvalsh(graphMatrixTime)
                        else:
                            W = np.linalg.eigvals(graphMatrixTime)
                        maxEigenvalue = np.max(np.real(W), axis = 1)
                        #   batchSize[b]
                        # Reshape to be able to divide by the graph matrix
                        maxEigenvalue=maxEigenvalue.reshape((batchSize[b],1,1))
                        # Normalize
                        graphMatrixTime = graphMatrixTime / maxEigenvalue
                                                              
                    # And put it in the corresponding time instant
                    graphMatrix[batchIndex[b]:batchIndex[b+1],t,:,:] = \
                                                                graphMatrixTime
                    
                    if doPrint:
                        # Sample percentage count
                        percentageCount = int(100*(t+1+b*tSamples)\
                                                          /(nBatches*tSamples))
                        
                        if t == 0 and b == 0:
                            # It's the first one, so just print it
                            print("%3d%%" % percentageCount,
                                  end = '', flush = True)
                        else:
                            # Erase the previous characters
                            print('\b \b' * 4 + "%3d%%" % percentageCount,
                                  end = '', flush = True)
                
            else:
                # Let's start by computing the distance squared
                _, distSq = self.computeDifferences(posBatch)
                # Apply the Kernel
                if kernelType == 'gaussian':
                    graphMatrixBatch = np.exp(-kernelScale * distSq)
                else:
                    graphMatrixBatch = distSq
                # Now let's place zeros in all places whose distance is greater
                # than the radius
                graphMatrixBatch[distSq > (commRadius ** 2)] = 0.
                # Set the diagonal elements to zero
                graphMatrixBatch[:,:,
                                 np.arange(0,nAgents),np.arange(0,nAgents)] =0.
                # If it is unweighted, force all nonzero values to be 1
                if not weighted:
                    graphMatrixBatch = (graphMatrixBatch > zeroTolerance)\
                                                          .astype(distSq.dtype)
                    
                if normalizeGraph:
                    isSymmetric = np.allclose(graphMatrixBatch,
                                              np.transpose(graphMatrixBatch,
                                                            axes = [0,1,3,2]))
                    # Tries to make the computation faster
                    if isSymmetric:
                        W = np.linalg.eigvalsh(graphMatrixBatch)
                    else:
                        W = np.linalg.eigvals(graphMatrixBatch)
                    maxEigenvalue = np.max(np.real(W), axis = 2)
                    #   batchSize[b] x tSamples
                    # Reshape to be able to divide by the graph matrix
                    maxEigenvalue = maxEigenvalue.reshape((batchSize[b],
                                                           tSamples,
                                                           1, 1))
                    # Normalize
                    graphMatrixBatch = graphMatrixBatch / maxEigenvalue
                    
                # Store
                graphMatrix[batchIndex[b]:batchIndex[b+1]] = graphMatrixBatch
                
                if doPrint:
                    # Sample percentage count
                    percentageCount = int(100*(b+1)/nBatches)
                    
                    if b == 0:
                        # It's the first one, so just print it
                        print("%3d%%" % percentageCount,
                              end = '', flush = True)
                    else:
                        # Erase the previous characters
                        print('\b \b' * 4 + "%3d%%" % percentageCount,
                              end = '', flush = True)
                    
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
            
        return graphMatrix
    
    def getData(self, name, samplesType, *args):
        
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
                    
        # Check that the name is actually an attribute
        assert name in dir(self)
        
        # Get the desired attribute
        thisDataDict = getattr(self, name)
        
        # Check it's a dictionary and that it has the corresponding key
        assert type(thisDataDict) is dict
        assert samplesType in thisDataDict.keys()
        
        # Get the data now
        thisData = thisDataDict[samplesType]
        # Get the dimension length
        thisDataDims = len(thisData.shape)
        
        # Check that it has at least two dimension, where the first one is
        # always the number of samples
        assert thisDataDims > 1
        
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = thisData.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # Select the corresponding samples
                thisData = thisData[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                thisData = thisData[args[0]]
                
            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(thisData.shape) < thisDataDims:
                if 'torch' in repr(thisData.dtype):
                    thisData =thisData.unsqueeze(0)
                else:
                    thisData = np.expand_dims(thisData, axis = 0)

        return thisData
        
    def evaluate(self, vel = None, accel = None, initVel = None,
                 samplingTime = None):
        
        # It is optional to add a different sampling time, if not, it uses
        # the internal one
        if samplingTime is None:
            # If there's no argument use the internal sampling time
            samplingTime = self.samplingTime
        
        # Check whether we have vel, or accel and initVel (i.e. we are either
        # given the velocities, or we are given the elements to compute them)
        if vel is not None:
            assert len(vel.shape) == 4
            nSamples = vel.shape[0]
            tSamples = vel.shape[1]
            assert vel.shape[2] == 2
            nAgents = vel.shape[3]
        elif accel is not None and initVel is not None:
            assert len(accel.shape) == 4 and len(initVel.shape) == 3
            nSamples = accel.shape[0]
            tSamples = accel.shape[1]
            assert accel.shape[2] == 2
            nAgents = accel.shape[3]
            assert initVel.shape[0] == nSamples
            assert initVel.shape[1] == 2
            assert initVel.shape[2] == nAgents
            
            # Now that we know we have a accel and init velocity, compute the
            # velocity trajectory
            # Compute the velocity trajectory
            if 'torch' in repr(accel.dtype):
                # Check that initVel is also torch
                assert 'torch' in repr(initVel.dtype)
                # Create the tensor to save the velocity trajectory
                vel = torch.zeros(nSamples,tSamples,2,nAgents,
                                  dtype = accel.dtype, device = accel.device)
                # Add the initial velocity
                vel[:,0,:,:] = initVel.clone().detach()
            else:
                # Create the space
                vel = np.zeros((nSamples, tSamples, 2, nAgents),
                               dtype=accel.dtype)
                # Add the initial velocity
                vel[:,0,:,:] = initVel.copy()
                
            # Go over time
            for t in range(1,tSamples):
                # Compute velocity
                vel[:,t,:,:] = accel[:,t-1,:,:] * samplingTime + vel[:,t-1,:,:]
            
        # Check that I did enter one of the if clauses
        assert vel is not None
            
        # And now that we have the velocities, we can compute the cost
        if 'torch' in repr(vel.dtype):
            # Average velocity for time t, averaged across agents
            avgVel = torch.mean(vel, dim = 3) # nSamples x tSamples x 2
            # Compute the difference in velocity between each agent and the
            # mean velocity
            diffVel = vel - avgVel.unsqueeze(3) 
            #   nSamples x tSamples x 2 x nAgents
            # Compute the MSE velocity
            diffVelNorm = torch.sum(diffVel ** 2, dim = 2) 
            #   nSamples x tSamples x nAgents
            # Average over agents
            diffVelAvg = torch.mean(diffVelNorm, dim = 2) # nSamples x tSamples
            # Sum over time
            costPerSample = torch.sum(diffVelAvg, dim = 1) # nSamples
            # Final average cost
            cost = torch.mean(costPerSample)
        else:
            # Repeat for numpy
            avgVel = np.mean(vel, axis = 3) # nSamples x tSamples x 2
            diffVel = vel - np.tile(np.expand_dims(avgVel, 3),
                                    (1, 1, 1, nAgents))
            #   nSamples x tSamples x 2 x nAgents
            diffVelNorm = np.sum(diffVel ** 2, axis = 2)
            #   nSamples x tSamples x nAgents
            diffVelAvg = np.mean(diffVelNorm, axis = 2) # nSamples x tSamples
            costPerSample = np.sum(diffVelAvg, axis = 1) # nSamples
            cost = np.mean(costPerSample) # scalar
        
        return cost
    
    def computeTrajectory(self, initPos, initVel, duration, **kwargs):
        
        # Check initPos is of shape batchSize x 2 x nAgents
        assert len(initPos.shape) == 3
        batchSize = initPos.shape[0]
        assert initPos.shape[1]
        nAgents = initPos.shape[2]
        
        # Check initVel is of shape batchSize x 2 x nAgents
        assert len(initVel.shape) == 3
        assert initVel.shape[0] == batchSize
        assert initVel.shape[1] == 2
        assert initVel.shape[2] == nAgents
        
        # Check what kind of data it is
        #   This is because all the functions are numpy, but if this was
        #   torch, we need to return torch, to make it consistent
        if 'torch' in repr(initPos.dtype):
            assert 'torch' in repr(initVel.dtype)
            useTorch = True
            device = initPos.device
            assert initVel.device == device
        else:
            useTorch = False
        
        # Create time line
        time = np.arange(0, duration, self.samplingTime)
        tSamples = len(time)
        
        # Here, we have two options, or we're given the acceleration or the
        # architecture
        assert 'archit' in kwargs.keys() or 'accel' in kwargs.keys()
        # Flags to determine which method to use
        useArchit = False
        useAccel = False
        
        if 'archit' in kwargs.keys():
            archit = kwargs['archit'] # This is a torch.nn.Module architecture
            architDevice = list(archit.parameters())[0].device
            useArchit = True
        elif 'accel' in kwargs.keys():
            accel = kwargs['accel']
            # accel has to be of shape batchSize x tSamples x 2 x nAgents
            assert len(accel.shape) == 4
            assert accel.shape[0] == batchSize
            assert accel.shape[1] == tSamples
            assert accel.shape[2] == 2
            assert accel.shape[3] == nAgents
            if useTorch:
                assert 'torch' in repr(accel.dtype)
            useAccel = True
            
        # Decide on printing or not:
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint # Use default
        
        # Now create the outputs that will be filled afterwards
        pos = np.zeros((batchSize, tSamples, 2, nAgents), dtype = np.float)
        vel = np.zeros((batchSize, tSamples, 2, nAgents), dtype = np.float)
        if useArchit:
            accel = np.zeros((batchSize, tSamples, 2, nAgents), dtype=np.float)
            state = np.zeros((batchSize, tSamples, 6, nAgents), dtype=np.float)
            graph = np.zeros((batchSize, tSamples, nAgents, nAgents),
                             dtype = np.float)
            
        # Assign the initial positions and velocities
        if useTorch:
            pos[:,0,:,:] = initPos.cpu().numpy()
            vel[:,0,:,:] = initVel.cpu().numpy()
            if useAccel:
                accel = accel.cpu().numpy()
        else:
            pos[:,0,:,:] = initPos.copy()
            vel[:,0,:,:] = initVel.copy()
            
        if doPrint:
            # Sample percentage count
            percentageCount = int(100/tSamples)
            # Print new value
            print("%3d%%" % percentageCount, end = '', flush = True)
            
        # Now, let's get started:
        for t in range(1, tSamples):
            
            # If it is architecture-based, we need to compute the state, and
            # for that, we need to compute the graph
            if useArchit:
                # Adjust pos value for graph computation
                thisPos = np.expand_dims(pos[:,t-1,:,:], 1)
                # Compute graph
                thisGraph = self.computeCommunicationGraph(thisPos,
                                                           self.commRadius,
                                                           True,
                                                           doPrint = False)
                # Save graph
                graph[:,t-1,:,:] = thisGraph.squeeze(1)
                # Adjust vel value for state computation
                thisVel = np.expand_dims(vel[:,t-1,:,:], 1)
                # Compute state
                thisState = self.computeStates(thisPos, thisVel, thisGraph,
                                               doPrint = False)
                # Save state
                state[:,t-1,:,:] = thisState.squeeze(1)
                
                # Compute the output of the architecture
                #   Note that we need the collection of all time instants up
                #   to now, because when we do the communication exchanges,
                #   it involves past times.
                x = torch.tensor(state[:,0:t,:,:], device = architDevice)
                S = torch.tensor(graph[:,0:t,:,:], device = architDevice)
                with torch.no_grad():
                    thisAccel = archit(x, S)
                # Now that we have computed the acceleration, we only care 
                # about the last element in time
                thisAccel = thisAccel.cpu().numpy()[:,-1,:,:]
                thisAccel[thisAccel > self.accelMax] = self.accelMax
                thisAccel[thisAccel < -self.accelMax] = self.accelMax
                # And save it
                accel[:,t-1,:,:] = thisAccel
                
            # Now that we have the acceleration, we can update position and
            # velocity
            vel[:,t,:,:] = accel[:,t-1,:,:] * self.samplingTime +vel[:,t-1,:,:]
            pos[:,t,:,:] = accel[:,t-1,:,:] * (self.samplingTime ** 2)/2 + \
                            vel[:,t-1,:,:] * self.samplingTime + pos[:,t-1,:,:]
            
            if doPrint:
                # Sample percentage count
                percentageCount = int(100*(t+1)/tSamples)
                # Erase previous value and print new value
                print('\b \b' * 4 + "%3d%%" % percentageCount,
                      end = '', flush = True)
                
        # And we're missing the last values of graph, state and accel, so
        # let's compute them for completeness
        #   Graph
        thisPos = np.expand_dims(pos[:,-1,:,:], 1)
        thisGraph = self.computeCommunicationGraph(thisPos, self.commRadius,
                                                   True, doPrint = False)
        graph[:,-1,:,:] = thisGraph.squeeze(1)
        #   State
        thisVel = np.expand_dims(vel[:,-1,:,:], 1)
        thisState = self.computeStates(thisPos, thisVel, thisGraph,
                                       doPrint = False)
        state[:,-1,:,:] = thisState.squeeze(1)
        #   Accel
        x = torch.tensor(state).to(architDevice)
        S = torch.tensor(graph).to(architDevice)
        with torch.no_grad():
            thisAccel = archit(x, S)
        thisAccel = thisAccel.cpu().numpy()[:,-1,:,:]
        thisAccel[thisAccel > self.accelMax] = self.accelMax
        thisAccel[thisAccel < -self.accelMax] = self.accelMax
        # And save it
        accel[:,-1,:,:] = thisAccel
                
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
            
        # After we have finished, turn it back into tensor, if required
        if useTorch:
            pos = torch.tensor(pos).to(device)
            vel = torch.tensor(vel).to(device)
            accel = torch.tensor(accel).to(device)
            
        # And return it
        if useArchit:
            return pos, vel, accel, state, graph
        elif useAccel:
            return pos, vel
    
    def computeDifferences(self, u):
        
        # Takes as input a tensor of shape
        #   nSamples x tSamples x 2 x nAgents
        # or of shape
        #   nSamples x 2 x nAgents
        # And returns the elementwise difference u_i - u_j of shape
        #   nSamples (x tSamples) x 2 x nAgents x nAgents
        # And the distance squared ||u_i - u_j||^2 of shape
        #   nSamples (x tSamples) x nAgents x nAgents
        
        # Check dimensions
        assert len(u.shape) == 3 or len(u.shape) == 4
        # If it has shape 3, which means it's only a single time instant, then
        # add the extra dimension so we move along assuming we have multiple
        # time instants
        if len(u.shape) == 3:
            u = np.expand_dims(u, 1)
            hasTimeDim = False
        else:
            hasTimeDim = True
        
        # Now we have that pos always has shape
        #   nSamples x tSamples x 2 x nAgents
        nSamples = u.shape[0]
        tSamples = u.shape[1]
        assert u.shape[2] == 2
        nAgents = u.shape[3]
        
        # Compute the difference along each axis. For this, we subtract a
        # column vector from a row vector. The difference tensor on each
        # position will have shape nSamples x tSamples x nAgents x nAgents
        # and then we add the extra dimension to concatenate and obtain a final
        # tensor of shape nSamples x tSamples x 2 x nAgents x nAgents
        # First, axis x
        #   Reshape as column and row vector, respectively
        uCol_x = u[:,:,0,:].reshape((nSamples, tSamples, nAgents, 1))
        uRow_x = u[:,:,0,:].reshape((nSamples, tSamples, 1, nAgents))
        #   Subtract them
        uDiff_x = uCol_x - uRow_x # nSamples x tSamples x nAgents x nAgents
        # Second, for axis y
        uCol_y = u[:,:,1,:].reshape((nSamples, tSamples, nAgents, 1))
        uRow_y = u[:,:,1,:].reshape((nSamples, tSamples, 1, nAgents))
        uDiff_y = uCol_y - uRow_y # nSamples x tSamples x nAgents x nAgents
        # Third, compute the distance tensor of shape
        #   nSamples x tSamples x nAgents x nAgents
        uDistSq = uDiff_x ** 2 + uDiff_y ** 2
        # Finally, concatenate to obtain the tensor of differences
        #   Add the extra dimension in the position
        uDiff_x = np.expand_dims(uDiff_x, 2)
        uDiff_y = np.expand_dims(uDiff_y, 2)
        #   And concatenate them
        uDiff = np.concatenate((uDiff_x, uDiff_y), 2)
        #   nSamples x tSamples x 2 x nAgents x nAgents
            
        # Get rid of the time dimension if we don't need it
        if not hasTimeDim:
            # (This fails if tSamples > 1)
            uDistSq = uDistSq.squeeze(1)
            #   nSamples x nAgents x nAgents
            uDiff = uDiff.squeeze(1)
            #   nSamples x 2 x nAgents x nAgents
            
        return uDiff, uDistSq
        
    def computeOptimalTrajectory(self, initPos, initVel, duration, 
                                 samplingTime, repelDist,
                                 accelMax = 100.):
        
        # The optimal trajectory is given by
        # u_{i} = - \sum_{j=1}^{N} (v_{i} - v_{j})
        #         + 2 \sum_{j=1}^{N} (r_{i} - r_{j}) *
        #                                 (1/\|r_{i}\|^{4} + 1/\|r_{j}\|^{2}) *
        #                                 1{\|r_{ij}\| < R}
        # for each agent i=1,...,N, where v_{i} is the velocity and r_{i} the
        # position.
        
        # Check that initPos and initVel as nSamples x 2 x nAgents arrays
        assert len(initPos.shape) == len(initVel.shape) == 3
        nSamples = initPos.shape[0]
        assert initPos.shape[1] == initVel.shape[1] == 2
        nAgents = initPos.shape[2]
        assert initVel.shape[0] == nSamples
        assert initVel.shape[2] == nAgents
        
        # time
        time = np.arange(0, duration, samplingTime)
        tSamples = len(time) # number of time samples
        
        # Create arrays to store the trajectory
        pos = np.zeros((nSamples, tSamples, 2, nAgents))
        vel = np.zeros((nSamples, tSamples, 2, nAgents))
        accel = np.zeros((nSamples, tSamples, 2, nAgents))
        
        # Initial settings
        pos[:,0,:,:] = initPos
        vel[:,0,:,:] = initVel
        
        if self.doPrint:
            # Sample percentage count
            percentageCount = int(100/tSamples)
            # Print new value
            print("%3d%%" % percentageCount, end = '', flush = True)
        
        # For each time instant
        for t in range(1,tSamples):
            
            # Compute the optimal acceleration
            #   Compute the distance between all elements (positions)
            ijDiffPos, ijDistSq = self.computeDifferences(pos[:,t-1,:,:])
            #       ijDiffPos: nSamples x 2 x nAgents x nAgents
            #       ijDistSq:  nSamples x nAgents x nAgents
            #   And also the difference in velocities
            ijDiffVel, _ = self.computeDifferences(vel[:,t-1,:,:])
            #       ijDiffVel: nSamples x 2 x nAgents x nAgents
            #   The last element we need to compute the acceleration is the
            #   gradient. Note that the gradient only counts when the distance 
            #   is smaller than the repel distance
            #       This is the mask to consider each of the differences
            repelMask = (ijDistSq < (repelDist**2)).astype(ijDiffPos.dtype)
            #       Apply the mask to the relevant differences
            ijDiffPos = ijDiffPos * np.expand_dims(repelMask, 1)
            #       Compute the constant (1/||r_ij||^4 + 1/||r_ij||^2)
            ijDistSqInv = invertTensorEW(ijDistSq)
            #       Add the extra dimension
            ijDistSqInv = np.expand_dims(ijDistSqInv, 1)
            #   Compute the acceleration
            accel[:,t-1,:,:] = \
                    -np.sum(ijDiffVel, axis = 3) \
                    +2* np.sum(ijDiffPos * (ijDistSqInv ** 2 + ijDistSqInv),
                               axis = 3)
                    
            # Finally, note that if the agents are too close together, the
            # acceleration will be very big to get them as far apart as
            # possible, and this is physically impossible.
            # So let's add a limitation to the maximum aceleration

            # Find the places where the acceleration is big
            thisAccel = accel[:,t-1,:,:].copy()
            # Values that exceed accelMax, force them to be accelMax
            thisAccel[accel[:,t-1,:,:] > accelMax] = accelMax
            # Values that are smaller than -accelMax, force them to be accelMax
            thisAccel[accel[:,t-1,:,:] < -accelMax] = -accelMax
            # And put it back
            accel[:,t-1,:,:] = thisAccel
            
            # Update the values
            #   Update velocity
            vel[:,t,:,:] = accel[:,t-1,:,:] * samplingTime + vel[:,t-1,:,:]
            #   Update the position
            pos[:,t,:,:] = accel[:,t-1,:,:] * (samplingTime ** 2)/2 + \
                                 vel[:,t-1,:,:] * samplingTime + pos[:,t-1,:,:]
            
            if self.doPrint:
                # Sample percentage count
                percentageCount = int(100*(t+1)/tSamples)
                # Erase previous pecentage and print new value
                print('\b \b' * 4 + "%3d%%" % percentageCount,
                      end = '', flush = True)
                
        # Print
        if self.doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
            
        return pos, vel, accel
        
    def computeInitialPositions(self, nAgents, nSamples, commRadius,
                                minDist = 0.1, geometry = 'rectangular',
                                **kwargs):
        
        # It will always be uniform. We can select whether it is rectangular
        # or circular (or some other shape) and the parameters respecting
        # that
        assert geometry == 'rectangular' or geometry == 'circular'
        assert minDist * (1.+zeroTolerance) <= commRadius * (1.-zeroTolerance)
        # We use a zeroTolerance buffer zone, just in case
        minDist = minDist * (1. + zeroTolerance)
        commRadius = commRadius * (1. - zeroTolerance)
        
        # If there are other keys in the kwargs argument, they will just be
        # ignored
        
        # We will first create the grid, whether it is rectangular or
        # circular.
        
        # Let's start by setting the fixed position
        if geometry == 'rectangular':
            
            # This grid has a distance that depends on the desired minDist and
            # the commRadius
            distFixed = (commRadius + minDist)/(2.*np.sqrt(2))
            #   This is the fixed distance between points in the grid
            distPerturb = (commRadius - minDist)/(4.*np.sqrt(2))
            #   This is the standard deviation of a uniform perturbation around
            #   the fixed point.
            # This should guarantee that, even after the perturbations, there
            # are no agents below minDist, and that all agents have at least
            # one other agent within commRadius.
            
            # How many agents per axis
            nAgentsPerAxis = int(np.ceil(np.sqrt(nAgents)))
            
            axisFixedPos = np.arange(-(nAgentsPerAxis * distFixed)/2,
                                       (nAgentsPerAxis * distFixed)/2,
                                      step = distFixed)
            
            # Repeat the positions in the same order (x coordinate)
            xFixedPos = np.tile(axisFixedPos, nAgentsPerAxis)
            # Repeat each element (y coordinate)
            yFixedPos = np.repeat(axisFixedPos, nAgentsPerAxis)
            
            # Concatenate this to obtain the positions
            fixedPos = np.concatenate((np.expand_dims(xFixedPos, 0),
                                       np.expand_dims(yFixedPos, 0)),
                                      axis = 0)
            
            # Get rid of unnecessary agents
            fixedPos = fixedPos[:, 0:nAgents]
            # And repeat for the number of samples we want to generate
            fixedPos = np.repeat(np.expand_dims(fixedPos, 0), nSamples,
                                 axis = 0)
            #   nSamples x 2 x nAgents
            
            # Now generate the noise
            perturbPos = np.random.uniform(low = -distPerturb,
                                           high = distPerturb,
                                           size = (nSamples, 2, nAgents))
            
            # Initial positions
            initPos = fixedPos + perturbPos
                
        elif geometry == 'circular':
            
            # Radius for the grid
            rFixed = (commRadius + minDist)/2.
            rPerturb = (commRadius - minDist)/4.
            fixedRadius = np.arange(0, rFixed * nAgents, step = rFixed)+rFixed
            
            # Angles for the grid
            aFixed = (commRadius/fixedRadius + minDist/fixedRadius)/2.
            for a in range(len(aFixed)):
                # How many times does aFixed[a] fits within 2pi?
                nAgentsPerCircle = 2 * np.pi // aFixed[a]
                # And now divide 2*np.pi by this number
                aFixed[a] = 2 * np.pi / nAgentsPerCircle
            #   Fixed angle difference for each value of fixedRadius
            
            # Now, let's get the radius, angle coordinates for each agents
            initRadius = np.empty((0))
            initAngles = np.empty((0))
            agentsSoFar = 0 # Number of agents located so far
            n = 0 # Index for radius
            while agentsSoFar < nAgents:
                thisRadius = fixedRadius[n]
                thisAngles = np.arange(0, 2*np.pi, step = aFixed[n])
                agentsSoFar += len(thisAngles)
                initRadius = np.concatenate((initRadius,
                                             np.repeat(thisRadius,
                                                       len(thisAngles))))
                initAngles = np.concatenate((initAngles, thisAngles))
                n += 1
                assert len(initRadius) == agentsSoFar
                
            # Restrict to the number of agents we need
            initRadius = initRadius[0:nAgents]
            initAngles = initAngles[0:nAgents]
            
            # Add the number of samples
            initRadius = np.repeat(np.expand_dims(initRadius, 0), nSamples,
                                   axis = 0)
            initAngles = np.repeat(np.expand_dims(initAngles, 0), nSamples,
                                   axis = 0)
            
            # Add the noise
            #   First, to the angles
            for n in range(nAgents):
                # Get the radius (the angle noise depends on the radius); so
                # far the radius is the same for all samples
                thisRadius = initRadius[0,n]
                aPerturb = (commRadius/thisRadius - minDist/thisRadius)/4.
                # Add the noise to the angles
                initAngles[:,n] += np.random.uniform(low = -aPerturb,
                                                     high = aPerturb,
                                                     size = (nSamples))
            #   Then, to the radius
            initRadius += np.random.uniform(low = -rPerturb,
                                            high = rPerturb,
                                            size = (nSamples, nAgents))
            
            # And finally, get the positions in the cartesian coordinates
            initPos = np.zeros((nSamples, 2, nAgents))
            initPos[:, 0, :] = initRadius * np.cos(initAngles)
            initPos[:, 1, :] = initRadius * np.sin(initAngles)
            
        # Now, check that the conditions are met:
        #   Compute square distances
        _, distSq = self.computeDifferences(np.expand_dims(initPos, 1))
        #   Get rid of the "time" dimension that arises from using the 
        #   method to compute distances
        distSq = distSq.squeeze(1)
        #   Compute the minimum distance (don't forget to add something in
        #   the diagonal, which otherwise is zero)
        minDistSq = np.min(distSq + \
                           2 * commRadius\
                             *np.eye(distSq.shape[1]).reshape(1,
                                                              distSq.shape[1],
                                                              distSq.shape[2])
                           )
        
        assert minDistSq >= minDist ** 2
        
        #   Now the number of neighbors
        graphMatrix = self.computeCommunicationGraph(np.expand_dims(initPos,1),
                                                     self.commRadius,
                                                     False,
                                                     doPrint = False)
        graphMatrix = graphMatrix.squeeze(1) # nSamples x nAgents x nAgents  
        
        #   Binarize the matrix
        graphMatrix = (np.abs(graphMatrix) > zeroTolerance)\
                                                         .astype(initPos.dtype)
        
        #   And check that we always have initially connected graphs
        for n in range(nSamples):
            assert graph.isConnected(graphMatrix[n,:,:])
        
        # We move to compute the initial velocities. Velocities can be
        # either positive or negative, so we do not need to determine
        # the lower and higher, just around zero
        if 'xMaxInitVel' in kwargs.keys():
            xMaxInitVel = kwargs['xMaxInitVel']
        else:
            xMaxInitVel = 3.
            #   Takes five seconds to traverse half the map
        # Same for the other axis
        if 'yMaxInitVel' in kwargs.keys():
            yMaxInitVel = kwargs['yMaxInitVel']
        else:
            yMaxInitVel = 3.
        
        # And sample the velocities
        xInitVel = np.random.uniform(low = -xMaxInitVel, high = xMaxInitVel,
                                     size = (nSamples, 1, nAgents))
        yInitVel = np.random.uniform(low = -yMaxInitVel, high = yMaxInitVel,
                                     size = (nSamples, 1, nAgents))
        # Add bias
        xVelBias = np.random.uniform(low = -xMaxInitVel, high = xMaxInitVel,
                                     size = (nSamples))
        yVelBias = np.random.uniform(low = -yMaxInitVel, high = yMaxInitVel,
                                     size = (nSamples))
        
        # And concatenate them
        velBias = np.concatenate((xVelBias, yVelBias)).reshape((nSamples,2,1))
        initVel = np.concatenate((xInitVel, yInitVel), axis = 1) + velBias
        #   nSamples x 2 x nAgents
        
        return initPos, initVel
        
        
    def saveVideo(self, saveDir, pos, *args, 
                  commGraph = None, **kwargs):
        
        # Check that pos is a position of shape nSamples x tSamples x 2 x nAgents
        assert len(pos.shape) == 4
        nSamples = pos.shape[0]
        tSamples = pos.shape[1]
        assert pos.shape[2] == 2
        nAgents = pos.shape[3]
        if 'torch' in repr(pos.dtype):
            pos = pos.cpu().numpy()
        
        # Check if there's the need to plot a graph
        if commGraph is not None:
            # If there's a communication graph, then it has to have shape
            #   nSamples x tSamples x nAgents x nAgents
            assert len(commGraph.shape) == 4
            assert commGraph.shape[0] == nSamples
            assert commGraph.shape[1] == tSamples
            assert commGraph.shape[2] == commGraph.shape[3] == nAgents
            if 'torch' in repr(commGraph.dtype):
                commGraph = commGraph.cpu().numpy()
            showGraph = True
        else:
            showGraph = False
        
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint
            
        # This number determines how faster or slower to reproduce the video
        if 'videoSpeed' in kwargs.keys():
            videoSpeed = kwargs['videoSpeed']
        else:
            videoSpeed = 1.
            
        if 'showVideoSpeed' in kwargs.keys():
            showVideoSpeed = kwargs['showVideoSpeed']
        else:
            if videoSpeed != 1:
                showVideoSpeed = True
            else:
                showVideoSpeed = False    
                
        if 'vel' in kwargs.keys():
            vel = kwargs['vel']
            if 'showCost' in kwargs.keys():
                showCost = kwargs['showCost']
            else:
                showCost = True
            if 'showArrows' in kwargs.keys():
                showArrows = kwargs['showArrows']
            else:
                showArrows = True
        else:
            showCost = False
            showArrows = False
        
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                selectedIndices = args[0]
                
            # Select the corresponding samples
            pos = pos[selectedIndices]
                
            # Finally, observe that if pos has shape only 3, then that's 
            # because we selected a single sample, so we need to add the extra
            # dimension back again
            if len(pos.shape) < 4:
                pos = np.expand_dims(pos, 0)
                
            if showGraph:
                commGraph = commGraph[selectedIndices]
                if len(commGraph.shape)< 4:
                    commGraph = np.expand_dims(commGraph, 0)
        
        # Where to save the video
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
            
        videoName = 'sampleTrajectory'
        
        xMinMap = np.min(pos[:,:,0,:]) * 1.2
        xMaxMap = np.max(pos[:,:,0,:]) * 1.2
        yMinMap = np.min(pos[:,:,1,:]) * 1.2
        yMaxMap = np.max(pos[:,:,1,:]) * 1.2
        
        # Create video object
        
        videoMetadata = dict(title = 'Sample Trajectory', artist = 'Flocking',
                             comment='Flocking example')
        videoWriter = FFMpegWriter(fps = videoSpeed/self.samplingTime,
                                   metadata = videoMetadata)
        
        if doPrint:
            print("\tSaving video(s)...", end = ' ', flush = True)
        
        # For each sample now
        for n in range(pos.shape[0]):
            
            # If there's more than one video to create, enumerate them
            if pos.shape[0] > 1:
                thisVideoName = videoName + '%03d.mp4' % n
            else:
                thisVideoName = videoName + '.mp4'
            
            # Select the corresponding position trajectory
            thisPos = pos[n]
            
            # Create figure
            videoFig = plt.figure(figsize = (5,5))
            
            # Set limits
            plt.xlim((xMinMap, xMaxMap))
            plt.ylim((yMinMap, yMaxMap))
            plt.axis('equal')
            
            if showVideoSpeed:
                plt.text(xMinMap, yMinMap, r'Speed: $%.2f$' % videoSpeed)
                
            # Create plot handle
            plotAgents, = plt.plot([], [], 
                                   marker = 'o',
                                   markersize = 3,
                                   linewidth = 0,
                                   color = '#01256E',
                                   scalex = False,
                                   scaley = False)
            
            # Create the video
            with videoWriter.saving(videoFig,
                                    os.path.join(saveDir,thisVideoName),
                                    tSamples):
                
                for t in range(tSamples):
                        
                    # Plot the agents
                    plotAgents.set_data(thisPos[t,0,:], thisPos[t,1,:])
                    videoWriter.grab_frame()
                    
                    # Print
                    if doPrint:
                        # Sample percentage count
                        percentageCount = int(
                                 100*(t+1+n*tSamples)/(tSamples * pos.shape[0])
                                              )
                        
                        if n == 0 and t == 0:
                            print("%3d%%" % percentageCount,
                                  end = '', flush = True)
                        else:
                            print('\b \b' * 4 + "%3d%%" % percentageCount,
                                  end = '', flush = True)
        
            plt.close(fig=videoFig)
            
        # Print
        if doPrint:
            # Erase the percentage and the label
            print('\b \b' * 4 + "OK", flush = True)
            
        if showGraph:
            
            # Normalize velocity
            if showArrows:
                # vel is of shape nSamples x tSamples x 2 x nAgents
                velNormSq = np.sum(vel ** 2, axis = 2)
                #   nSamples x tSamples x nAgents
                maxVelNormSq = np.max(np.max(velNormSq, axis = 2), axis = 1)
                #   nSamples
                maxVelNormSq = maxVelNormSq.reshape((nSamples, 1, 1, 1))
                #   nSamples x 1 x 1 x 1
                normVel = 2*vel/np.sqrt(maxVelNormSq)
            
            if doPrint:
                print("\tSaving graph snapshots...", end = ' ', flush = True)
            
            # Essentially, we will print nGraphs snapshots and save them
            # as images with the graph. This is the best we can do in a
            # reasonable processing time (adding the graph to the video takes
            # forever).
            time = np.arange(0, self.duration, step = self.samplingTime)
            assert len(time) == tSamples
            
            nSnapshots = 5 # The number of snapshots we will consider
            tSnapshots = np.linspace(0, tSamples-1, num = nSnapshots)
            #   This gives us nSnapshots equally spaced in time. Now, we need
            #   to be sure these are integers
            tSnapshots = np.unique(tSnapshots.astype(np.int)).astype(np.int)
            
            # Directory to save the snapshots
            snapshotDir = os.path.join(saveDir,'graphSnapshots')
            # Base name of the snapshots
            snapshotName = 'graphSnapshot'
            
            for n in range(pos.shape[0]):
                
                if pos.shape[0] > 1:
                    thisSnapshotDir = snapshotDir + '%03d' % n
                    thisSnapshotName = snapshotName + '%03d' % n
                else:
                    thisSnapshotDir = snapshotDir
                    thisSnapshotName = snapshotName
                    
                if not os.path.exists(thisSnapshotDir):
                    os.mkdir(thisSnapshotDir)
                
                # Get the corresponding positions
                thisPos = pos[n]
                thisCommGraph = commGraph[n]
                
                for t in tSnapshots:
                    
                    # Get the edge pairs
                    #   Get the graph for this time instant
                    thisCommGraphTime = thisCommGraph[t]
                    #   Check if it is symmetric
                    isSymmetric = np.allclose(thisCommGraphTime,
                                              thisCommGraphTime.T)
                    if isSymmetric:
                        #   Use only half of the matrix
                        thisCommGraphTime = np.triu(thisCommGraphTime)
                    
                    #   Find the position of all edges
                    outEdge, inEdge = np.nonzero(np.abs(thisCommGraphTime) \
                                                               > zeroTolerance)
                    
                    # Create the figure
                    thisGraphSnapshotFig = plt.figure(figsize = (5,5))
                    
                    # Set limits (to be the same as the video)
                    plt.xlim((xMinMap, xMaxMap))
                    plt.ylim((yMinMap, yMaxMap))
                    plt.axis('equal')
                    
                    # Plot the edges
                    plt.plot([thisPos[t,0,outEdge], thisPos[t,0,inEdge]],
                             [thisPos[t,1,outEdge], thisPos[t,1,inEdge]],
                             color = '#A8AAAF', linewidth = 0.75,
                             scalex = False, scaley = False)
                    
                    # Plot the arrows
                    if showArrows:
                        for i in range(nAgents):
                            plt.arrow(thisPos[t,0,i], thisPos[t,1,i],
                                      normVel[n,t,0,i], normVel[n,t,1,i])
                
                    # Plot the nodes
                    plt.plot(thisPos[t,0,:], thisPos[t,1,:],
                             marker = 'o', markersize = 3, linewidth = 0,
                             color = '#01256E', scalex = False, scaley = False)
                    
                    # Add the cost value
                    if showCost:
                        totalCost = self.evaluate(vel = vel[:,t:t+1,:,:])
                        plt.text(xMinMap,yMinMap, r'Cost: $%.4f$' % totalCost)
                    
                    # Add title
                    plt.title("Time $t=%.4f$s" % time[t])
                    
                    # Save figure
                    thisGraphSnapshotFig.savefig(os.path.join(thisSnapshotDir,
                                            thisSnapshotName + '%03d.pdf' % t))
                    
                    # Close figure
                    plt.close(fig = thisGraphSnapshotFig)
                    
                    # Print percentage completion
                    if doPrint:
                        # Sample percentage count
                        percentageCount = int(
                                 100*(t+1+n*tSamples)/(tSamples * pos.shape[0])
                                              )
                        if n == 0 and t == 0:
                            # Print new value
                            print("%3d%%" % percentageCount,
                                  end = '', flush = True)
                        else:
                            # Erase the previous characters
                            print('\b \b' * 4 + "%3d%%" % percentageCount,
                                  end = '', flush = True)
                        
                        
                
            # Print
            if doPrint:
                # Erase the percentage and the label
                print('\b \b' * 4 + "OK", flush = True)
            
class TwentyNews(_dataForClassification):
    """
    TwentyNews: Loads and handles handles the 20NEWS dataset

    Initialization:

    Input:
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        nWords (int): number of words to consider (i.e. the nWords most frequent
            words in the news articles are kept, the rest, discarded)
        nWordsShortDocs (int): any article with less words than nWordsShortDocs
            are discarded.
        nEdges (int): how many edges to keep after creating a geometric graph
            considering the graph embedding of each new article.
        distMetric (string): function to use to compute the distance between
            articles in the embedded space.
        dataDir (string): directory where to download the 20News dataset to/
            to check if it has already been downloaded
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:
        
    .getData(dataSubset): loads the data belonging to dataSubset (i.e. 'train' 
        or 'test')
    
    .embedData(): compute the graph embedding of the training dataset after
        it has been loaded
    
    .normalizeData(normType): normalize the data in the embedded space following
        a normType norm.
    
    .createValidationSet(ratio): stores ratio% of the training set as validation
        set.
        
    .createGraph(): uses the word2vec embedding of the training set to compute
        a geometric graph
        
    .getGraph(): fetches the adjacency matrix of the stored graph
    
    .getNumberOfClasses(): fetches the number of classes
    
    .reduceDataset(nTrain, nValid, nTest): reduces the dataset by randomly
        selected nTrain, nValid and nTest samples from the training, validation
        and testing datasets, respectively.
        
    authorData = .getAuthorData(samplesType, selectData, [, optionalArguments])
    
        Input:
            samplesType (string): 'train', 'valid', 'test' or 'all' to determine
                from which dataset to get the raw author data from
            selectData (string): 'WAN' or 'wordFreq' to decide if we want to
                retrieve either the WAN of each excerpt or the word frequency
                count of each excerpt
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        
        Output:
            Either the WANs or the word frequency count of all the excerpts of
            the selected author
            
    .createGraph(): creates a graph from the WANs of the excerpt written by the
        selected author available in the training set. The fusion of this WANs
        is done in accordance with the input options following 
        graphTools.createGraph().
        The resulting adjacency matrix is stored.
        
    .getGraph(): fetches the stored adjacency matrix and returns it
    
    .getFunctionWords(): fetches the list of functional words. Returns a tuple
        where the first element correspond to all the functional words in use, 
        and the second element consists of all the functional words available.
        Obs.: When we created the graph, some of the functional words might have
        been dropped in order to make it connected, for example.

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """
    def __init__(self, ratioValid, nWords, nWordsShortDocs, nEdges, distMetric,
                 dataDir, dataType = np.float64, device = 'cpu'):
        
        super().__init__()
        # This creates the attributes: dataType, device, nTrain, nTest, nValid,
        # and samples, and fills them all with None, and also creates the 
        # methods: getSamples, astype, to, and evaluate.
        self.dataType = dataType
        self.device = device
        
        # Other relevant information we need to store:
        self.dataDir = dataDir # Where the data is
        self.N = nWords # Number of nodes
        self.nWordsShortDocs = nWordsShortDocs # Number of words under which
            # a document is too short to be taken into consideration
        self.M = nEdges # Number of edges
        self.distMetric = distMetric # Distance metric to use
        self.dataset = {} # Here we save the dataset classes as they are
            # handled by mdeff's code
        self.nClasses = None # Number of classes
        self.vocab = None # Words considered
        self.graphData = None # Store the data (word2vec embeddings) required
            # to build the graph
        self.adjacencyMatrix = None # Store the graph built from the loaded
            # data
    
        # Get the training dataset. Saves vocab, dataset, and samples
        self.getData('train')
        # Embeds the data following the N words and a word2vec approach, saves
        # the embedded vectors in graphData, and updates vocab to keep only
        # the N words selected
        self.embedData()
        # Get the testing dataset, only for the words stored in vocab.
        self.getData('test')
        # Normalize
        self.normalizeData()
        # Save number of samples
        self.nTrain = self.samples['train']['targets'].shape[0]
        self.nTest = self.samples['test']['targets'].shape[0]
        # Create validation set
        self.createValidationSet(ratioValid)
        # Create graph
        self.createGraph() # Only after data has been embedded
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def getData(self, dataSubset):
        
        # Load dataset
        dataset = Text20News(data_home = self.dataDir,
                             subset = dataSubset,
                             remove = ('headers','footers','quotes'),
                             shuffle = True)
        # Get rid of numbers and other stuff
        dataset.clean_text(num='substitute')
        # If there's some vocabulary already defined, vectorize (count the
        # frequencies) of the words in vocab, if not, count all of them
        if self.vocab is None:
            dataset.vectorize(stop_words='english')
            self.vocab = dataset.vocab
        else:
            dataset.vectorize(vocabulary = self.vocab)
    
        # Get rid of short documents
        if dataSubset == 'train':
            dataset.remove_short_documents(nwords = self.nWordsShortDocs,
                                           vocab = 'full')
            # Get rid of images
            dataset.remove_encoded_images()
            self.nClasses = len(dataset.class_names)
        else:
            dataset.remove_short_documents(nwords = self.nWordsShortDocs,
                                           vocab = 'selected')
        
        # Save them in the corresponding places
        self.samples[dataSubset]['signals'] = dataset.data.toarray()
        self.samples[dataSubset]['targets'] = dataset.labels
        self.dataset[dataSubset] = dataset
        
    def embedData(self):
        
        # We need to have loaded the training dataset first.
        assert 'train' in self.dataset.keys()
        # Embed them (word2vec embedding)
        self.dataset['train'].embed()
        # Keep only the top words (which determine the number of nodes)
        self.dataset['train'].keep_top_words(self.N)
        # Update the vocabulary
        self.vocab = self.dataset['train'].vocab
        # Get rid of short documents when considering only the specific 
        # vocabulary
        self.dataset['train'].remove_short_documents(
                                                  nwords = self.nWordsShortDocs,
                                                  vocab = 'selected')
        # Save the embeddings, which are necessary to build a graph
        self.graphData = self.dataset['train'].embeddings
        # Update the samples
        self.samples['train']['signals'] = self.dataset['train'].data.toarray()
        self.samples['train']['targets'] = self.dataset['train'].labels
        # If there's an existing dataset, update it to the new vocabulary
        if 'test' in self.dataset.keys():
            self.dataset['test'].vectorize(vocabulary = self.vocab)
            # Update the samples
            self.samples['test']['signals'] =self.dataset['test'].data.toarray()
            self.samples['test']['targets'] = self.dataset['test'].labels
        
    def normalizeData(self, normType = 'l1'):
        
        for key in self.dataset.keys():
            # Normalize the frequencies on the l1 norm.
            self.dataset[key].normalize(norm = normType)
            # And save it
            self.samples[key]['signals'] = self.dataset[key].data.toarray()
            self.samples[key]['targets'] = self.dataset[key].labels
            
    def createValidationSet(self, ratio):
        # How many valid samples
        self.nValid = int(ratio * self.nTrain)
        # Shuffle indices
        randomIndices = np.random.permutation(self.nTrain)
        validationIndices = randomIndices[0:self.nValid]
        trainIndices = randomIndices[self.nValid:]
        # Fetch those samples and put them in the validation set
        self.samples['valid']['signals'] = self.samples['train']['signals']\
                                                          [validationIndices, :]
        self.samples['valid']['targets'] = self.samples['train']['targets']\
                                                             [validationIndices]
        # And update the training set
        self.samples['train']['signals'] = self.samples['train']['signals']\
                                                               [trainIndices, :]
        self.samples['train']['targets'] = self.samples['train']['targets']\
                                                                  [trainIndices]
        # Update the numbers
        self.nValid = self.samples['valid']['targets'].shape[0]
        self.nTrain = self.samples['train']['targets'].shape[0]
            
    def createGraph(self, *args):
        
        assert self.graphData is not None
        assert len(args) == 0 or len(args) == 2
        if len(args) == 2:
            self.M = args[0] # Number of edges
            self.distMetric = args[1] # Distance metric
        dist, idx = distance_sklearn_metrics(self.graphData, k = self.M,
                                             metric = self.distMetric)
        self.adjacencyMatrix = adjacency(dist, idx).toarray()
        
    def getGraph(self):
        
        return self.adjacencyMatrix
    
    def getNumberOfClasses(self):
        
        return self.nClasses
    
    def reduceDataset(self, nTrain, nValid, nTest):
        if nTrain < self.nTrain:
            randomIndices = np.random.permutation(self.nTrain)
            trainIndices = randomIndices[0:nTrain]
            # And update the training set
            self.samples['train']['signals'] = self.samples['train']\
                                                           ['signals']\
                                                           [trainIndices, :]
            self.samples['train']['targets'] = self.samples['train']\
                                                           ['targets']\
                                                           [trainIndices]
            self.nTrain = nTrain
        if nValid < self.nValid:
            randomIndices = np.random.permutation(self.nValid)
            validIndices = randomIndices[0:nValid]
            # And update the training set
            self.samples['valid']['signals'] = self.samples['valid']\
                                                           ['signals']\
                                                           [validIndices, :]
            self.samples['valid']['targets'] = self.samples['valid']\
                                                           ['targets']\
                                                           [validIndices]
            self.nValid = nValid
        if nTest < self.nTest:
            randomIndices = np.random.permutation(self.nTest)
            testIndices = randomIndices[0:nTest]
            # And update the training set
            self.samples['test']['signals'] = self.samples['test']\
                                                           ['signals']\
                                                           [testIndices, :]
            self.samples['test']['targets'] = self.samples['test']\
                                                          ['targets']\
                                                          [testIndices]
            self.nTest = nTest
    
    def astype(self, dataType):
        # This changes the type for the graph data, as well as the adjacency
        # matrix. We are going to leave the dataset attribute as it is, since
        # this is the most accurate reflection of mdeff's code.
        self.graphData = changeDataType(self.graphData, dataType)
        self.adjacencyMatrix = changeDataType(self.adjacencyMatrix, dataType)

        # And now, initialize to change the samples as well (and also save the 
        # data type)
        super().astype(dataType)
        
    
    def to(self, device):
        # If the dataType is 'torch'
        if repr(self.dataType).find('torch') >= 0:
            # Change the stored attributes that are not handled by the inherited
            # method to().
            self.graphData.to(device)
            self.adjacencyMatrix.to(device)
            # And call the inherit method to initialize samples (and save to
            # device)
            super().to(device)
    
# Copied almost verbatim from the code by Michäel Defferrard, available at
# http://github.com/mdeff/cnn_graph

import gensim
import sklearn, sklearn.datasets, sklearn.metrics
import scipy.sparse

import re

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
	"""Compute exact pairwise distances."""
	d = sklearn.metrics.pairwise.pairwise_distances(
			z, metric=metric)
	# k-NN graph.
	idx = np.argsort(d)[:, 1:k+1]
	d.sort()
	d = d[:, 1:k+1]
	return d, idx

def adjacency(dist, idx):
	"""Return the adjacency matrix of a kNN graph."""
	M, k = dist.shape
	assert M, k == idx.shape
	assert dist.min() >= 0

	# Weights.
	sigma2 = np.mean(dist[:, -1])**2
	dist = np.exp(- dist**2 / sigma2)

	# Weight matrix.
	I = np.arange(0, M).repeat(k)
	J = idx.reshape(M*k)
	V = dist.reshape(M*k)
	W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

	# No self-connections.
	W.setdiag(0)

	# Non-directed graph.
	bigger = W.T > W
	W = W - W.multiply(bigger) + W.T.multiply(bigger)

	assert W.nnz % 2 == 0
	assert np.abs(W - W.T).mean() < 1e-10
	assert type(W) is scipy.sparse.csr.csr_matrix
	return W

def replace_random_edges(A, noise_level):
	"""Replace randomly chosen edges by random edges."""
	M, M = A.shape
	n = int(noise_level * A.nnz // 2)

	indices = np.random.permutation(A.nnz//2)[:n]
	rows = np.random.randint(0, M, n)
	cols = np.random.randint(0, M, n)
	vals = np.random.uniform(0, 1, n)
	assert len(indices) == len(rows) == len(cols) == len(vals)

	A_coo = scipy.sparse.triu(A, format='coo')
	assert A_coo.nnz == A.nnz // 2
	assert A_coo.nnz >= n
	A = A.tolil()

	for idx, row, col, val in zip(indices, rows, cols, vals):
		old_row = A_coo.row[idx]
		old_col = A_coo.col[idx]

		A[old_row, old_col] = 0
		A[old_col, old_row] = 0
		A[row, col] = 1
		A[col, row] = 1

	A.setdiag(0)
	A = A.tocsr()
	A.eliminate_zeros()
	return A
        
class TextDataset(object):
    def clean_text(self, num='substitute'):
        # TODO: stemming, lemmatisation
        for i,doc in enumerate(self.documents):
            # Digits.
            if num == 'spell':
                doc = doc.replace('0', ' zero ')
                doc = doc.replace('1', ' one ')
                doc = doc.replace('2', ' two ')
                doc = doc.replace('3', ' three ')
                doc = doc.replace('4', ' four ')
                doc = doc.replace('5', ' five ')
                doc = doc.replace('6', ' six ')
                doc = doc.replace('7', ' seven ')
                doc = doc.replace('8', ' eight ')
                doc = doc.replace('9', ' nine ')
            elif num == 'substitute':
                # All numbers are equal. Useful for embedding
                # (countable words) ?
                doc = re.sub('(\\d+)', ' NUM ', doc)
            elif num == 'remove':
                # Numbers are uninformative (they are all over the place).
                # Useful for bag-of-words ?
                # But maybe some kind of documents contain more numbers,
                # e.g. finance.
                # Some documents are indeed full of numbers. At least
                # in 20NEWS.
                doc = re.sub('[0-9]', ' ', doc)
            # Remove everything except a-z characters and single space.
            doc = doc.replace('$', ' dollar ')
            doc = doc.lower()
            doc = re.sub('[^a-z]', ' ', doc)
            doc = ' '.join(doc.split()) # same as 
                                        # doc = re.sub('\s{2,}', ' ', doc)
            self.documents[i] = doc

    def vectorize(self, **params):
        # TODO: count or tf-idf. Or in normalize ?
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(**params)
        self.data = vectorizer.fit_transform(self.documents)
        self.vocab = vectorizer.get_feature_names()
        assert len(self.vocab) == self.data.shape[1]

    def keep_documents(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.documents = [self.documents[i] for i in idx]
        self.labels = self.labels[idx]
        self.data = self.data[idx,:]

    def keep_words(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.data = self.data[:,idx]
        self.vocab = [self.vocab[i] for i in idx]
        try:
            self.embeddings = self.embeddings[idx,:]
        except AttributeError:
            pass

    def remove_short_documents(self, nwords, vocab='selected'):
        """Remove a document if it contains less than nwords."""
        if vocab == 'selected':
            # Word count with selected vocabulary.
            wc = self.data.sum(axis=1)
            wc = np.squeeze(np.asarray(wc))
        elif vocab == 'full':
            # Word count with full vocabulary.
            wc = np.empty(len(self.documents), dtype=np.int)
            for i,doc in enumerate(self.documents):
                wc[i] = len(doc.split())
        idx = np.argwhere(wc >= nwords).squeeze()
        self.keep_documents(idx)

    def keep_top_words(self, M):
        """Keep in the vocaluary the M words who appear most often."""
        freq = self.data.sum(axis=0)
        freq = np.squeeze(np.asarray(freq))
        idx = np.argsort(freq)[::-1]
        idx = idx[:M]
        self.keep_words(idx)

    def normalize(self, norm='l1'):
        """Normalize data to unit length."""
        # TODO: TF-IDF.
        data = self.data.astype(np.float64)
        self.data = sklearn.preprocessing.normalize(data, axis=1, norm=norm)

    def embed(self, filename=None, size=100):
        """Embed the vocabulary using pre-trained vectors."""
        if filename:
            model = gensim.models.Word2Vec.load_word2vec_format(filename,
                                                                binary=True)
            size = model.vector_size
        else:
            class Sentences(object):
                def __init__(self, documents):
                    self.documents = documents
                def __iter__(self):
                    for document in self.documents:
                        yield document.split()
            model = gensim.models.Word2Vec(Sentences(self.documents), size=size)
        self.embeddings = np.empty((len(self.vocab), size))
        keep = []
        not_found = 0
        for i,word in enumerate(self.vocab):
            try:
                self.embeddings[i,:] = model[word]
                keep.append(i)
            except KeyError:
                not_found += 1
        self.keep_words(keep)
        
    def remove_encoded_images(self, freq=1e3):
        widx = self.vocab.index('ax')
        wc = self.data[:,widx].toarray().squeeze()
        idx = np.argwhere(wc < freq).squeeze()
        self.keep_documents(idx)

class Text20News(TextDataset):
    def __init__(self, **params):
        dataset = sklearn.datasets.fetch_20newsgroups(**params)
        self.documents = dataset.data
        self.labels = dataset.target
        self.class_names = dataset.target_names
        assert max(self.labels) + 1 == len(self.class_names)

class Epidemics(_data):
# Luana R. Ruiz, rubruiz@seas.upenn.edu, 2021/03/04
        def __init__(self, seqLen, seedProb, infectionProb, recoveryTime, 
                     nTrain, nValid, nTest, x0 = None, dataType = np.float64, 
                     device = 'cpu'):
            
            super().__init__()
            self.seqLen = seqLen
            self.seedProb = seedProb
            self.infectionProb = infectionProb
            self.recoveryTime = recoveryTime
            self.nTrain = nTrain
            self.nValid = nValid
            self.nTest = nTest
            nSamples = nTrain + nValid + nTest
            self.dataType = dataType
            self.device = device
            
            self.Adj = self.createGraph()
            self.N = self.Adj.shape[0]
            
            if x0 is not None:
                self.x0 = x0
            else: 
                x0 = np.random.binomial(1,self.seedProb,(nSamples,self.N))
                while np.sum(np.sum(x0,axis=1)>0) < nSamples:
                    x0 = np.random.binomial(1,self.seedProb,(nSamples,self.N))
                self.x0 = x0
                
            horizon = 2*seqLen
            x_t = x0
            x = np.expand_dims(x_t, axis=1)
            timeInfection = np.zeros((self.N,nSamples))
            for t in range(1,horizon):
                x_tplus1 = x_t
                for n in range(nSamples):
                    for i in range(self.N):
                        if x_t[n,i] == 1:
                            for j in list(np.argwhere(self.Adj[i,i:]>0)):
                                if x_t[n,j] == 0:
                                    x_tplus1[n,j] == np.random.binomial(1,
                                                    infectionProb*t/horizon)
                                    timeInfection[j,n] = t
                            if timeInfection[i,n]-t >= recoveryTime:
                                x_tplus1[n,i] = 2
                x_t = x_tplus1    
                x = np.concatenate((x,np.expand_dims(x_t, axis=1)),axis=1)
                
            y = x[:,seqLen:horizon,:] == 1
            x = x[:,:seqLen,:] 
            
            self.samples['train']['signals'] = x[0:nTrain,:,:]
            self.samples['train']['targets'] = y[0:nTrain,:,:]
            self.samples['valid']['signals'] = x[nTrain:nTrain+nValid,:,:]
            self.samples['valid']['targets'] = y[nTrain:nTrain+nValid,:,:]
            self.samples['test']['signals'] = x[nTrain+nValid:nSamples,:,:]
            self.samples['test']['targets'] = y[nTrain+nValid:nSamples,:,:]
                
        @staticmethod
        def createGraph():
            
            edge_list = []
            with open('datasets/epidemics/edge_list.txt') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                for row in csv_reader:
                    aux_list = []
                    aux_list.append(int(row[0])-1)
                    aux_list.append(int(row[1])-1)
                    edge_list.append(aux_list)
            nNodes = max(max(edge_list))+1
            Adj = [[0]*nNodes for _ in range(nNodes)]
            for sink, source in edge_list:
                Adj[sink][source] = 1
            Adj = np.array(Adj)
            Adj = Adj + np.transpose(Adj) > 0
            idx_0 = np.argwhere(np.matmul(Adj,np.ones(nNodes))>0).squeeze()
            Adj = Adj[idx_0,:]
            Adj = Adj[:,idx_0]
            
            return Adj
        
        def evaluate(self, yHat, y, tol = 1e-9):
            
            dimensions = len(yHat.shape)
            C = yHat.shape[dimensions-2]
            N = yHat.shape[dimensions-1]
            yHat = yHat.reshape((-1,C,N))
            yHat = torch.nn.functional.log_softmax(yHat, dim=1)
            yHat = torch.exp(yHat)
            yHat = torch.argmax(yHat,dim=1)
            yHat = yHat.double()
            y = y.reshape((-1,N))
            
            tp = torch.sum(y*yHat,1)
            #tn = torch.sum((1-y)*(1-yHat),1)
            fp = torch.sum((1-y)*yHat,1)
            fn = torch.sum(y*(1-yHat),1)
        
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            
            idx_p = p!=p
            idx_tp = tp<tol
            idx_p1 = idx_p*idx_tp
            p[idx_p] = 0
            p[idx_p1] = 1
            idx_r = r!=r
            idx_r1 = idx_r*idx_tp
            r[idx_r] = 0
            r[idx_r1] = 1
        
            f1 = 2*p*r / (p+r)
            f1[f1!=f1] = 0
            
            return 1 - torch.mean(f1)
                    
            
