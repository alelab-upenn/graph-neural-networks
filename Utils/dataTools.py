# 2018/12/04~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
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
"""

import os
import pickle
import hdf5storage # This is required to import old Matlab(R) files.
import urllib.request # To download from the internet
import zipfile # To handle zip files
import gzip # To handle gz files
import shutil # Command line utilities

import numpy as np
import torch

import Utils.graphTools as graph

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
        # type: train, valid, test
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
        # type: train, valid, test
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
            workingMatrix = self.incompleteMatrix
            # In this case, each row is a user (how that user scored all movies)
            # and this is the kind of data samples we need for movie-based
            # graphs
            indexNodesToKeep = indexColsToKeep
        
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
            thisSignal = self.samples[key]['signals'] # B x N
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
