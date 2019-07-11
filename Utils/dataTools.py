# 2018/12/4~~2018/07/12
# Fernando Gama, fgama@seas.upenn.edu
"""
dataTools.py Data management module

Several tools to manage data

FacebookEgo (class): loads the Facebook adjacency matrix of EgoNets
SourceLocalization (class): creates the datasets for a source localization 
    problem
Authorship (class): loads and splits the dataset for the authorship attribution
    problem
MovieLens (class): Loads and handles handles the MovieLens-100k dataset
TwentyNews (class): handles the 20NEWS dataset
"""
## IMPORTANT NOTE (gensim): The 20NEWS dataset relies on the gensim library.
# I have found several issues with this library, so in this release, the 
# importing of this library has been commented. Please, uncomment it, and be
# sure to have it installed before using the 20NEWS dataset.
# (The importing line is located below the TwentyNews class, where all the
# auxiliary functions are defined)

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

class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   getSamples(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
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
        self.samples['train']['labels'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['labels'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['labels'] = None
        
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
        y = self.samples[samplesType]['labels']
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
                # requested
                x = x[selectedIndices,:].reshape([args[0], x.shape[1]])
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                x = x[args[0], :]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(x.shape) == 1:
                    x = x.reshape([1, x.shape[0]])
                # And assign the labels
                y = y[args[0]]

        return x, y

    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        
        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers. To do this we need to
        # match the desired dataType to its int counterpart. Typical examples
        # are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32
        
        labelType = str(self.samples['train']['labels'].dtype)
        if 'int' in labelType:
            if 'numpy' in repr(dataType) or 'np' in repr(dataType):
                if '64' in labelType:
                    labelType = np.int64
                elif '32' in labelType:
                    labelType = np.int32
            elif 'torch' in repr(dataType):
                if '64' in labelType:
                    labelType = torch.int64
                elif '32' in labelType:
                    labelType = torch.int32
        else: # If there is no int, just stick with the given dataType
            labelType = dataType
        
        # Now that we have selected the dataType, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        if 'torch' in repr(dataType): # If it is torch
            for key in self.samples.keys():
                self.samples[key]['signals'] = \
                       torch.tensor(self.samples[key]['signals']).type(dataType)
                self.samples[key]['labels'] = \
                       torch.tensor(self.samples[key]['labels']).type(labelType)
        else: # If it is not torch
            for key in self.samples.keys():
                self.samples[key]['signals'] = \
                                          dataType(self.samples[key]['signals'])
                self.samples[key]['labels'] = \
                                          labelType(self.samples[key]['labels'])

        # Update attribute
        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if repr(self.dataType).find('torch') >= 0:
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
    # cases (how many examples are correctly labels) so justifies the use of
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
            accuracy = 1 - totalErrors.type(self.dataType)/N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis = 1)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
        return accuracy
        
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
        signals = x # nTotal x N (CS notation)
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.
        nodesToLabels = {}
        for it in range(len(sourceNodes)):
            nodesToLabels[sourceNodes[it]] = it
        labels = [nodesToLabels[x] for x in sampledSources] # nTotal
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['labels'] = np.array(labels[0:nTrain])
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['labels'] = np.array(labels[nTrain:nTrain+nValid])
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['labels'] = np.array(labels[nTrain+nValid:nTotal])
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
    
class Authorship(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

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
        self.samples['train']['labels'] = labelsTrain.astype(np.int)
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['labels'] = labelsValid.astype(np.int)
        self.samples['test']['signals'] = xTest
        self.samples['test']['labels'] = labelsTest.astype(np.int)
        # Create graph
        self.createGraph()
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that 
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
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
        if self.samples['train']['signals'] is not None:
            self.samples['train']['signals'] = \
                                self.samples['train']['signals'][:, nodesToKeep]
        if self.samples['valid']['signals'] is not None:
            self.samples['valid']['signals'] = \
                                self.samples['valid']['signals'][:, nodesToKeep]
        if self.samples['test']['signals'] is not None:
            self.samples['test']['signals'] = \
                                self.samples['test']['signals'][:, nodesToKeep]
        if self.allFunctionWords is not None:
            self.functionWords = [self.allFunctionWords[w] for w in nodesToKeep]
        
    def getGraph(self):
        
        return self.adjacencyMatrix
    
    def getFunctionWords(self):
        
        return self.functionWords, self.allFunctionWords
    
    def astype(self, dataType):
        # This changes the type for the selected author as well as the samples
        # First, the selected author info
        if repr(dataType).find('torch') == -1:
            for key in self.selectedAuthor.keys():
                for secondKey in self.selectedAuthor[key].keys():
                    self.selectedAuthor[key][secondKey] \
                                 = dataType(self.selectedAuthor[key][secondKey])
            self.adjacencyMatrix = dataType(self.adjacencyMatrix)
        else:
            for key in self.selectedAuthor.keys():
                for secondKey in self.selectedAuthor[key].keys():
                    self.selectedAuthor[key][secondKey] \
                            = torch.tensor(self.selectedAuthor[key][secondKey])\
                                    .type(dataType)
            self.adjacencyMatrix = torch.tensor(self.adjacencyMatrix)\
                                        .type(dataType)

        # And now, initialize to change the samples as well (and also save the 
        # data type)
        super().astype(dataType)
        
    
    def to(self, device):
        # If the dataType is 'torch'
        if repr(self.dataType).find('torch') >= 0:
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
        The setting is that of regression on a single node of the graph. That
        is, given a graph, and an incomplete graph signal on that graph, we want
        to estimate the value of the signal on a single node.
        
        If, for instance, we have a movie-based graph, then the graph signal
        corresponds to the ratings that a given user gave to some of the movies.
        The objective is to estimate how that particular user would rate one
        of the other available movies. (Same holds by interchanging 'movie' with
        'user' in this paragraph)
        
        This differs from the classical setting. One typical way of addressing
        this problem is as matrix completion: estimate the value that _all_
        user would give to _all_ movies. Another typical setting is to estimate
        the value that a single user would give to _all_ movies (or how a single
        movie would be rated by _all_ users).
        
        Obs.: The loadData() method that builds the incomplete matrix with
        the data base would still work if this is to be used for one of the more
        standard problems.

    Initialization:

    Input:
        graphType('user' or 'movie'): which underlying graph to build; 'user'
            for user-based graph (each node is a user), and 'movie' for 
            movie-based (each node is a movie); this also determines the data,
            on a user-based graph, each data sample corresponds to a movie, and
            on the movie-based graph, each data sample corresponds to a user.
        labelID (int): this specific node is selected to be interpolated; 
            this has effect in the building of the training, validation and 
            test sets, since only data samples that have a value
            at that node can be used
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
        categorical (bool): if the data is going to be estimated as a
            categorical variable (default: True; since there are only a finite
            number of integer ratings, we can solve this regression problem
            as a classification problem in which we want to guess what is the
            'category' the estimated rating falls in, if this is the case, then
            the evaluate() method needs to convert the one-hot vector into
            an actual rating, before computing the RMSE --i.e., even though the
            problem can be solved by a classification problem, the evaluation
            function is still the RMSE which is the standard use)
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:
        
    .loadData(filename, [dataDir]): loads the data from dataDir (if not
        provided, the internally stored one is used) and saves it as filename;
        if the data has already been processed and saved as 'filename', then
        it will be just loaded.
        
    .createGraph(): creates a graphType-based graph with the previously
        established options (undirected, isolated, connected, etc.); this graph 
        is always sparsified by means of a nearest-neighbor algorithm.
    
    .getGraph(): fetches the adjacency matrix of the stored graph
    
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
        and the actual ratings given in y; if categorical = True, then yHat
        is expected to be a vector of size the number of possible ratings, and
        the value of the rating is taken to be the position where the maximum
        value of that vector is.

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    lossValue = .evaluate(yHat, y)
        Input:
            yHat (dtype.array): estimated target
            y (dtype.array): target representation

        Output:
            lossValue (float): regression loss chosen

    """
    
    def __init__(self, graphType, labelID, ratioTrain, ratioValid, dataDir,
                 keepIsolatedNodes, forceUndirected, forceConnected, kNN,
                 categorical = True, dataType = np.float64, device = 'cpu'):
        
        super().__init__()
        # This creates the attributes: dataType, device, nTrain, nTest, nValid,
        # and samples, and fills them all with None, and also creates the 
        # methods: getSamples, astype, to, and evaluate.
        self.dataType = dataType
        self.device = device
        
        # Store attributes
        #   GraphType
        assert graphType == 'user' or graphType == 'movie'
        # This is because what are the graph signals depends on the graph we
        # want to use.
        self.graphType = graphType
        #   Label ID
        assert labelID > 0
        self.labelID = labelID - 1 # -1 is to match the movieLens indexing 
        #   (which) starts at 1, with the row/column indexing, which starts at
        #   zero and is the one we will actually use (is to avoid subtracting 
        #   -1 everywhere else)
        # Label ID is the user ID or the movie ID following the MovieLens 
        # nomenclature. This determines how we build the labels in the
        # dataset. If it's all, then we are in the regression problem, and
        # the "label" is just the entire signal. [Not sure this would fit in the
        # framework].
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
        #   Evaluation processing
        self.categorical = categorical
        #   Empty attributes for now
        self.incompleteMatrix = None
        self.movieTitles = {}
        self.adjacencyMatrix = None
        self.indexDataPoints = {}
        
        # Now, we should be ready to load the data and build the (incomplete) 
        # matrix
        self.loadData('movielens100kIncompleteMatrix.pkl')
        # This has loaded the incompleteMatrix and movieTitles attributes.
        
        # To simplify code, we will work always with each row being a data
        # sample. The incompleteMatrix is User x Movies
        if graphType == 'user':
            # If the graph type is user-based, then the graph signals are the
            # movies, scored for every user. This means that each column of the
            # incompleteMatrix is a graph signal, but since we're working with
            # rows, we have to transpose it
            workingMatrix = self.incompleteMatrix.T # Movies x User
            # Now, each row is a movie score for all users, so that it is a
            # graph signal in the user-based graph.
        else:
            workingMatrix = self.incompleteMatrix
            # In this case, each row is a user (how that user scored all movies)
            # and this is the kind of data samples we need for movie-based
            # graphs
            
        # Now, let's determine the labelID so we can then do the split.
        # Basically, if labelID not 'all', then the number of datasamples
        # will depend on how many are available for that ID. If it's all, then
        # they just all go there.
        
        # If we want to select a specific column, we first, need to check
        # that the labelID is one of the columns (since the columns index 
        # the nodes, and we want to know the estimate the value of the 
        # signal at a single node, we need to be sure that this node is
        # there)
        assert self.labelID < workingMatrix.shape[1]
        # Extract that column (recall that movieLens IDs start at 1)
        selectedID = workingMatrix[:, self.labelID]
        # Now we have to check the number of nonzero elements in the column
        indexDataPoints = np.nonzero(selectedID)[0]
        # This is because np.nonzero() returns a tuple for the index of the
        # nonzero elements in each dimension, but here, when we select a
        # column, we have only one dimension.
        nDataPoints = len(indexDataPoints) # Total number of points
        # Now we split all the valid points into train/validation/test
        self.nTrain = round(ratioTrain * nDataPoints) # Total train set
        self.nValid = round(ratioValid * self.nTrain) # Validation set
        self.nTrain = self.nTrain - self.nValid # Effective train set
        self.nTest = nDataPoints - self.nTrain - self.nValid
        # Permute the indices at random
        randPerm = np.random.permutation(nDataPoints)
        # And choose the indices that will correspond to each dataset
        indexTrainPoints = indexDataPoints[randPerm[0:self.nTrain]]
        indexValidPoints = indexDataPoints[\
                            randPerm[self.nTrain : self.nTrain+self.nValid]]
        indexTestPoints = indexDataPoints[\
                            randPerm[self.nTrain+self.nValid : nDataPoints]]
        # Finally get the corresponding samples and store them
        self.samples['train']['signals'] = workingMatrix[indexTrainPoints,:]
        self.samples['train']['labels'] = selectedID[indexTrainPoints]
        self.samples['valid']['signals'] = workingMatrix[indexValidPoints,:]
        self.samples['valid']['labels'] = selectedID[indexValidPoints]
        self.samples['test']['signals'] = workingMatrix[indexTestPoints,:]
        self.samples['test']['labels'] = selectedID[indexTestPoints]
        # And update the index of the data points.
        self.indexDataPoints['all'] = indexDataPoints
        self.indexDataPoints['train'] = indexTrainPoints
        self.indexDataPoints['valid'] = indexValidPoints
        self.indexDataPoints['test'] = indexTestPoints
            
        # Now the data has been loaded, and the training/test partition has been
        # made, create the graph
        self.createGraph()
        
        # Now that all the empty elements with zeros have been dealt with (and
        # the self.incompleteMatrix still has zeros), if we are having
        # categorical variables, then we need to force this to be labels
        # starting at 0 (and ints)
        if self.categorical:
            if '64' in str(self.samples['train']['signals'].dtype):
                labelDataType = np.int64 # At this point, everything is still in
                    # numpy
            elif '32' in str(self.samples['train']['signals'].dtype):
                labelDataType = np.int32 # At this point, everything is still in
                    # numpy
            self.samples['train']['labels'] = \
                              labelDataType(self.samples['train']['labels'] - 1)
            self.samples['valid']['labels'] = \
                              labelDataType(self.samples['valid']['labels'] - 1)
            self.samples['test']['labels'] = \
                               labelDataType(self.samples['test']['labels'] - 1)
        
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
        # taken into account.
        # To follow the paper by Huang et al., where the data is given by
        # Y in U x M, and goes into full detail on how to build the U x U
        # user-based graph, then, we will stick with this formulation
        if self.graphType == 'user':
            workingMatrix = self.incompleteMatrix # User x Movies
        else:
            workingMatrix = self.incompleteMatrix.T # Movies x User
        # Note that this is the opposite arrangement that we considered before
        # when loading the data into samples; back then we considered samples
        # to be rows and the data to build the graph was therefore in columns;
        # in this case, it is the opposite, since we still want to use the data
        # located in the rows.
        
        # Now, no matter what we choose, every row corresponds to a node in the
        # graph.
        
        # Now we need to partition the dataset into a train set, but be sure
        # that we included the samples that were used for the training set in
        # the self.samples.
        
        # So most of what comes down here is how to select the samples that
        # we already selected as train set, and complement that with other 
        # data samples. The problem is that the rest of the samples we
        # are picking them at random from the entire incomplete matrix, where
        # the ones we already have are for one specific row (column in the
        # loading data).
        
        # The number of data points is given by the number of nonzero elements 
        # of the matrix
        indexDataPoints = np.nonzero(workingMatrix)
        # This is a tuple, where the first element is the place of nonzero
        # indices in the rows, and the second element is the place of nonzero
        # indices in the columns.
        nDataPoints = len(indexDataPoints[0]) # or [1], it doesn't matter
        # Note that every nonzero point belonging to labelID has already been
        # assigned to either one or the other dataset, so when we split
        # these datasets, we cannot consider these.
        
        # Let's start with computing how many training points we still need
        nTrainPointsAll = round(self.ratioTrain * nDataPoints)
        nTrainPointsRest = nTrainPointsAll - self.nTrain - self.nValid
        # This is the number of points we still need.
        
        # Now, indexDataPoints have all nTrainPointsAll non zero elements,
        # we need to discard those that have labelID in the rows
        subIndexDataPointsRestAll = np.nonzero(indexDataPoints[0] \
                                                             != self.labelID)[0]
        #   Nonzero points along the rows
        indexDataPointsRestAll = (indexDataPoints[0][subIndexDataPointsRestAll],
                                  indexDataPoints[1][subIndexDataPointsRestAll])
        #   Update the indices (rows and columns) of all the data points to
        #   exclude those with labelID in the rows
        nDataPointsRest = nDataPoints - self.nTrain - self.nTest - self.nValid
        #   Number of data points left
        assert len(indexDataPointsRestAll[0]) == nDataPointsRest
        #   To check that we picked the right number of elements
        
        # Now, get the random permutation of these elements
        randPerm = np.random.permutation(nDataPointsRest)
        # Pick the number needed for training
        subIndexRandomRest = randPerm[0:nTrainPointsRest]
        # And select the necessary ones at random
        indexTrainPointsRest = (indexDataPointsRestAll[0][subIndexRandomRest],
                                indexDataPointsRestAll[1][subIndexRandomRest])
        # So far, we made it to select the appropriate number of training 
        # samples, at random, that do not include the ones we already selected
        # according to labelID.
        
        # Now, we need to join this selected training points from the rest of 
        # the dataset, with the points from the labelID training set
        #   Get the points from the labelID (both training and valid)
        allTrainPointsID = np.concatenate((self.indexDataPoints['train'],
                                           self.indexDataPoints['valid']))
        #   We need to add the label ID in the rows, to fit them in the
        #   context of the matrix
        labelIDpadding = self.labelID * np.ones(self.nTrain + self.nValid)
        allTrainPointsID = (labelIDpadding.astype(allTrainPointsID.dtype),
                            allTrainPointsID)
        #   And join them to the actual points
        indexTrainPoints = (
                   np.concatenate((indexTrainPointsRest[0],allTrainPointsID[0])),
                   np.concatenate((indexTrainPointsRest[1],allTrainPointsID[1]))
                            )
        # And this is it! We got all the necessary training samples, including
        # those that we were already using.
        
        # Finally, set every other element not in the training set in the 
        # workingMatrix to zero
        workingMatrixZeroedTrain = workingMatrix.copy()      
        workingMatrixZeroedTrain[indexTrainPoints] = 0
        workingMatrix = workingMatrix - workingMatrixZeroedTrain
        assert len(np.nonzero(workingMatrix)[0]) == nTrainPointsAll 
        # To check that the total number of nonzero elements of the matrix are
        # the total number of training samples that we're supposed to have.
        
        # Now, we finally have the incompleteMatrix only with the corresponding
        # elements: a ratioTrain proportion of training samples that, for sure,
        # include the ones that we will use in the graph signals dataset
       
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
        
        # Finally, normalize the individual user variances and get ride of the
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
        assert self.labelID not in isolatedNodes[0]
        #   It is highly likely that the labelID won't be isolated node, since 
        #   we are purposefully taking training samples that include this item,
        #   but just in case.
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
        # However, if there are more connected components, maybe the labelID
        # is not part of the largest.
        if self.labelID not in nodesToKeep:
        #   So, this is interesting: we know that labelID is not part of the
        #   isolated nodes, because we check that before. Therefore, if it
        #   is not part of the largest component, it has to be part of some 
        #   other component. This implies that we do not need to check if
        #   extraComponents is nonempty: it HAS to be nonempty.
            assert len(extraComponents) > 0
            # Check in each component if the labelID node is there
            for n in range(len(extraComponents)):
                # If it is there
                if self.labelID in extraComponents[1][n]:
                    # Save the corresponding adjacency matrix
                    W = extraComponents[0][n]
                    # And the corresponding list of nodes to keep
                    nodesToKeep = extraComponents[1][n]
        #   Update samples and labelID, if necessary
        if len(nodesToKeep) < N:
            # Update labelID
            self.labelID = nodesToKeep.index(self.labelID)
            # Update samples
            if self.samples['train']['signals'] is not None:
                self.samples['train']['signals'] = \
                                self.samples['train']['signals'][:, nodesToKeep]
            if self.samples['valid']['signals'] is not None:
                self.samples['valid']['signals'] = \
                                self.samples['valid']['signals'][:, nodesToKeep]
            if self.samples['test']['signals'] is not None:
                self.samples['test']['signals'] = \
                                self.samples['test']['signals'][:, nodesToKeep]
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
        
    def getIncompleteMatrix(self):
        
        return self.incompleteMatrix
    
    def getGraph(self):
        
        return self.adjacencyMatrix
    
    def getMovieTitles(self):
        
        return self.movieTitles
    
    def getLabelID(self):
        
        return self.labelID
    
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
        
        #If it is a categorical variable
        if self.categorical:
            # We need to convert the categories into the corresponding hardmax
            if 'torch' in repr(self.dataType):
                #   We compute the target label (hardmax)
                yHat = torch.argmax(yHat, dim = 1).type(self.dataType)
                y = y.type(self.dataType)
            else:
                yHat = np.array(yHat)
                y = np.array(y)
                #   We compute the target label (hardmax)
                yHat = np.argmax(yHat, axis = 1).astype(yHat.dtype)
                y = y.astype(yHat.dtype)
        
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
        if repr(dataType).find('torch') == -1:
            self.incompleteMatrix = dataType(self.incompleteMatrix)
            self.adjacencyMatrix = dataType(self.adjacencyMatrix)
        else:
            self.incompleteMatrix = torch.tensor(self.incompleteMatrix)\
                                                                 .type(dataType)
            self.adjacencyMatrix = torch.tensor(self.adjacencyMatrix)\
                                                                 .type(dataType)

        # And now, initialize to change the samples as well (and also save the 
        # data type)
        super().astype(dataType)
        
    
    def to(self, device):
        # If the dataType is 'torch'
        if repr(self.dataType).find('torch') >= 0:
            # Change the stored attributes that are not handled by the inherited
            # method to().
            self.incompleteMatrix.to(device)
            self.adjacencyMatrix.to(device)
            # And call the inherit method to initialize samples (and save to
            # device)
            super().to(device)
            
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
        self.nTrain = self.samples['train']['labels'].shape[0]
        self.nTest = self.samples['test']['labels'].shape[0]
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
        self.samples[dataSubset]['labels'] = dataset.labels
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
        self.samples['train']['labels'] = self.dataset['train'].labels
        # If there's an existing dataset, update it to the new vocabulary
        if 'test' in self.dataset.keys():
            self.dataset['test'].vectorize(vocabulary = self.vocab)
            # Update the samples
            self.samples['test']['signals'] =self.dataset['test'].data.toarray()
            self.samples['test']['labels'] = self.dataset['test'].labels
        
    def normalizeData(self, normType = 'l1'):
        
        for key in self.dataset.keys():
            # Normalize the frequencies on the l1 norm.
            self.dataset[key].normalize(norm = normType)
            # And save it
            self.samples[key]['signals'] = self.dataset[key].data.toarray()
            self.samples[key]['labels'] = self.dataset[key].labels
            
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
        self.samples['valid']['labels'] = self.samples['train']['labels']\
                                                             [validationIndices]
        # And update the training set
        self.samples['train']['signals'] = self.samples['train']['signals']\
                                                               [trainIndices, :]
        self.samples['train']['labels'] = self.samples['train']['labels']\
                                                                  [trainIndices]
        # Update the numbers
        self.nValid = self.samples['valid']['labels'].shape[0]
        self.nTrain = self.samples['train']['labels'].shape[0]
            
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
            self.samples['train']['labels'] = self.samples['train']\
                                                          ['labels']\
                                                          [trainIndices]
            self.nTrain = nTrain
        if nValid < self.nValid:
            randomIndices = np.random.permutation(self.nValid)
            validIndices = randomIndices[0:nValid]
            # And update the training set
            self.samples['valid']['signals'] = self.samples['valid']\
                                                           ['signals']\
                                                           [validIndices, :]
            self.samples['valid']['labels'] = self.samples['valid']\
                                                          ['labels']\
                                                          [validIndices]
            self.nValid = nValid
        if nTest < self.nTest:
            randomIndices = np.random.permutation(self.nTest)
            testIndices = randomIndices[0:nTest]
            # And update the training set
            self.samples['test']['signals'] = self.samples['test']\
                                                           ['signals']\
                                                           [testIndices, :]
            self.samples['test']['labels'] = self.samples['test']\
                                                          ['labels']\
                                                          [testIndices]
            self.nTest = nTest
    
    def astype(self, dataType):
        # This changes the type for the graph data, as well as the adjacency
        # matrix. We are going to leave the dataset attribute as it is, since
        # this is the most accurate reflection of mdeff's code.
        if repr(dataType).find('torch') == -1:
            self.graphData = dataType(self.graphData)
            self.adjacencyMatrix = dataType(self.adjacencyMatrix)
        else:
            self.graphData = torch.tensor(self.graphData).type(dataType)
            self.adjacencyMatrix = torch.tensor(self.adjacencyMatrix)\
                                        .type(dataType)

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

#import gensim
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
            if num is 'spell':
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
            elif num is 'substitute':
                # All numbers are equal. Useful for embedding
                # (countable words) ?
                doc = re.sub('(\\d+)', ' NUM ', doc)
            elif num is 'remove':
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
        if vocab is 'selected':
            # Word count with selected vocabulary.
            wc = self.data.sum(axis=1)
            wc = np.squeeze(np.asarray(wc))
        elif vocab is 'full':
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
