# 2018/12/03~
# Fernando Gama, fgama@seas.upenn.edu.
# Luana Ruiz, rubruiz@seas.upenn.edu.
"""
graphTools.py Tools for handling graphs

Functions:

plotGraph: plots a graph from an adjacency matrix
printGraph: prints (saves) a graph from an adjacency matrix
adjacencyToLaplacian: transform an adjacency matrix into a Laplacian matrix
normalizeAdjacency: compute the normalized adjacency
normalizeLaplacian: compute the normalized Laplacian
computeGFT: Computes the eigenbasis of a GSO
matrixPowers: computes the matrix powers
computeNonzeroRows: compute nonzero elements across rows
computeNeighborhood: compute the neighborhood of a graph
computeSourceNodes: compute source nodes for the source localization problem
isConnected: determines if a graph is connected
sparsifyGraph: sparsifies a given graph matrix
createGraph: creates an adjacency marix
permIdentity: identity permutation
permDegree: order nodes by degree
permSpectralProxies: order nodes by spectral proxies score
permEDS: order nodes by EDS score
edgeFailSampling: samples the edges of a given graph
splineBasis: Returns the B-spline basis (taken from github.com/mdeff)

Classes:

Graph: class containing a graph
"""

import numpy as np
import scipy.sparse
import scipy.spatial as sp
from sklearn.cluster import SpectralClustering

import os
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt

zeroTolerance = 1e-9 # Values below this number are considered zero.

# If adjacency matrices are not symmetric these functions might not work as
# desired: the degree will be the in-degree to each node, and the Laplacian
# is not defined for directed graphs. Same caution is advised when using
# graphs with self-loops.

def plotGraph(adjacencyMatrix, **kwargs):
    """
    plotGraph(A): plots a graph from adjacency matrix A of size N x N
    
    Optional keyword arguments:
        'positions' (np.array, default: points in a circle of radius 1):
                size N x 2 of positions for each node
        'figSize' (int, default: 5): size of the figure
        'linewidth' (int, default: 1): edge width
        'markerSize' (int, default: 15): node size
        'markerShape' (string, default: 'o'): node shape
        'color' (hex code string, default: '#01256E'): color of the nodes
        'nodeLabel' (list, default: None): list of length N where each element
            corresponds to the label of each node
    """
    
    # Data
    #   Adjacency matrix
    W = adjacencyMatrix
    assert W.shape[0] == W.shape[1]
    N = W.shape[0]
    #   Positions (optional)
    if 'positions' in kwargs.keys():
        pos = kwargs['positions']
    else:
        angle = np.linspace(0, 2*np.pi*(1-1/N), num = N)
        radius = 1
        pos = np.array([
                radius * np.sin(angle),
                radius * np.cos(angle)
                ])
        
    # Create figure
    #   Figure size
    if 'figSize' in kwargs.keys():
        figSize = kwargs['figSize']
    else:
        figSize = 5
    #   Line width
    if 'lineWidth' in kwargs.keys():
        lineWidth = kwargs['lineWidth']
    else:
        lineWidth = 1
    #   Marker Size
    if 'markerSize' in kwargs.keys():
        markerSize = kwargs['markerSize']
    else:
        markerSize = 15
    #   Marker shape
    if 'markerShape' in kwargs.keys():
        markerShape = kwargs['markerShape']
    else:
        markerShape = 'o'
    #   Marker color
    if 'color' in kwargs.keys():
        markerColor = kwargs['color']
    else:
        markerColor = '#01256E'
    #   Node labeling
    if 'nodeLabel' in kwargs.keys():
        doText = True
        nodeLabel = kwargs['nodeLabel']
        assert len(nodeLabel) == N
    else:
        doText = False
        
    # Plot
    figGraph = plt.figure(figsize = (1*figSize, 1*figSize))
    for i in range(N):
        for j in range(N):
            if W[i,j] > 0:
                plt.plot([pos[0,i], pos[0,j]], [pos[1,i], pos[1,j]],
                         linewidth = W[i,j] * lineWidth,
                         color = '#A8AAAF')
    for i in range(N):
        plt.plot(pos[0,i], pos[1,i], color = markerColor,
                 marker = markerShape, markerSize = markerSize)
        if doText:
            plt.text(pos[0,i], pos[1,i], nodeLabel[i],
                     verticalalignment = 'center',
                     horizontalalignment = 'center',
                     color = '#F2F2F3')
            
    return figGraph

def printGraph(adjacencyMatrix, **kwargs):
    """
    printGraph(A): Wrapper for plot graph to directly save it as a graph (with 
        no axis, nor anything else like that, more aesthetic, less changes)
    
    Optional keyword arguments:
        'saveDir' (os.path, default: '.'): directory where to save the graph
        'legend' (default: None): Text for a legend
        'xLabel' (str, default: None): Text for the x axis
        'yLabel' (str, default: None): Text for the y axis
        'graphName' (str, default: 'graph'): name to save the file
        'positions' (np.array, default: points in a circle of radius 1):
                size N x 2 of positions for each node
        'figSize' (int, default: 5): size of the figure
        'linewidth' (int, default: 1): edge width
        'markerSize' (int, default: 15): node size
        'markerShape' (string, default: 'o'): node shape
        'color' (hex code string, default: '#01256E'): color of the nodes
        'nodeLabel' (list, default: None): list of length N where each element
            corresponds to the label of each node
    """
    
    # Wrapper for plot graph to directly save it as a graph (with no axis,
    # nor anything else like that, more aesthetic, less changes)
    
    W = adjacencyMatrix
    assert W.shape[0] == W.shape[1]
    
    # Printing options
    if 'saveDir' in kwargs.keys():
        saveDir = kwargs['saveDir']
    else:
        saveDir = '.'
    if 'legend' in kwargs.keys():
        doLegend = True
        legendText = kwargs['legend']
    else:
        doLegend = False
    if 'xLabel' in kwargs.keys():
        doXlabel = True
        xLabelText = kwargs['xLabel']
    else:
        doXlabel = False
    if 'yLabel' in kwargs.keys():
        doYlabel = True
        yLabelText = kwargs['yLabel']
    else:
        doYlabel = False
    if 'graphName' in kwargs.keys():
        graphName = kwargs['graphName']
    else:
        graphName = 'graph'
    
    figGraph = plotGraph(adjacencyMatrix, **kwargs)
    
    plt.axis('off')
    if doXlabel:
        plt.xlabel(xLabelText)
    if doYlabel:
        plt.yLabel(yLabelText)
    if doLegend:
        plt.legend(legendText)
    
    figGraph.savefig(os.path.join(saveDir, '%s.pdf' % graphName),
                     bbox_inches = 'tight', transparent = True)

def adjacencyToLaplacian(W):
    """
    adjacencyToLaplacian: Computes the Laplacian from an Adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        L (np.array): Laplacian matrix
    """
    # Check that the matrix is square
    assert W.shape[0] == W.shape[1]
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # And build the degree matrix
    D = np.diag(d)
    # Return the Laplacian
    return D - W

def normalizeAdjacency(W):
    """
    NormalizeAdjacency: Computes the degree-normalized adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        A (np.array): degree-normalized adjacency matrix
    """
    # Check that the matrix is square
    assert W.shape[0] == W.shape[1]
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # Invert the square root of the degree
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Adjacency
    return D @ W @ D

def normalizeLaplacian(L):
    """
    NormalizeLaplacian: Computes the degree-normalized Laplacian matrix

    Input:

        L (np.array): Laplacian matrix

    Output:

        normL (np.array): degree-normalized Laplacian matrix
    """
    # Check that the matrix is square
    assert L.shape[0] == L.shape[1]
    # Compute the degree vector (diagonal elements of L)
    d = np.diag(L)
    # Invert the square root of the degree
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Laplacian
    return D @ L @ D

def computeGFT(S, order = 'no'):
    """
    computeGFT: Computes the frequency basis (eigenvectors) and frequency
        coefficients (eigenvalues) of a given GSO

    Input:

        S (np.array): graph shift operator matrix
        order (string): 'no', 'increasing', 'totalVariation' chosen order of
            frequency coefficients (default: 'no')

    Output:

        E (np.array): diagonal matrix with the frequency coefficients
            (eigenvalues) in the diagonal
        V (np.array): matrix with frequency basis (eigenvectors)
    """
    # Check the correct order input
    assert order == 'totalVariation' or order == 'no' or order == 'increasing'
    # Check the matrix is square
    assert S.shape[0] == S.shape[1]
    # Check if it is symmetric
    symmetric = np.allclose(S, S.T, atol = zeroTolerance)
    # Then, compute eigenvalues and eigenvectors
    if symmetric:
        e, V = np.linalg.eigh(S)
    else:
        e, V = np.linalg.eig(S)
    # Sort the eigenvalues by the desired error:
    if order == 'totalVariation':
        eMax = np.max(e)
        sortIndex = np.argsort(np.abs(e - eMax))
    elif order == 'increasing':
        sortIndex = np.argsort(np.abs(e))
    else:
        sortIndex = np.arange(0, S.shape[0])
    e = e[sortIndex]
    V = V[:, sortIndex]
    E = np.diag(e)
    return E, V

def matrixPowers(S,K):
    """
    matrixPowers(A, K) Computes the matrix powers A^k for k = 0, ..., K-1

    Inputs:
        A: either a single N x N matrix or a collection E x N x N of E matrices.
        K: integer, maximum power to be computed (up to K-1)

    Outputs:
        AK: either a collection of K matrices K x N x N (if the input was a
            single matrix) or a collection E x K x N x N (if the input was a
            collection of E matrices).
    """
    # S can be either a single GSO (N x N) or a collection of GSOs (E x N x N)
    if len(S.shape) == 2:
        N = S.shape[0]
        assert S.shape[1] == N
        E = 1
        S = S.reshape(1, N, N)
        scalarWeights = True
    elif len(S.shape) == 3:
        E = S.shape[0]
        N = S.shape[1]
        assert S.shape[2] == N
        scalarWeights = False

    # Now, let's build the powers of S:
    thisSK = np.tile(np.eye(N, N).reshape(1,N,N), [E, 1, 1])
    SK = thisSK.reshape(E, 1, N, N)
    for k in range(1,K):
        thisSK = thisSK @ S
        SK = np.concatenate((SK, thisSK.reshape(E, 1, N, N)), axis = 1)
    # Take out the first dimension if it was a single GSO
    if scalarWeights:
        SK = SK.reshape(K, N, N)

    return SK

def computeNonzeroRows(S, Nl = 'all'):
    """
    computeNonzeroRows: Find the position of the nonzero elements of each
        row of a matrix

    Input:

        S (np.array): matrix
        Nl (int or 'all'): number of rows to compute the nonzero elements; if
            'all', then Nl = S.shape[0]. Rows are counted from the top.

    Output:

        nonzeroElements (list): list of size Nl where each element is an array
            of the indices of the nonzero elements of the corresponding row.
    """
    # Find the position of the nonzero elements of each row of the matrix S.
    # Nl = 'all' means for all rows, otherwise, it will be an int.
    if Nl == 'all':
        Nl = S.shape[0]
    assert Nl <= S.shape[0]
    # Save neighborhood variable
    neighborhood = []
    # For each of the selected nodes
    for n in range(Nl):
        neighborhood += [np.flatnonzero(S[n,:])]

    return neighborhood

def computeNeighborhood(S, K, N = 'all', nb = 'all', outputType = 'list'):
    """
    computeNeighborhood: compute the set of nodes within the K-hop neighborhood
        of a graph (i.e. all nodes that can be reached within K-hops of each
        node)

        computeNeighborhood(W, K, N = 'all', nb = 'all', outputType = 'list')

    Input:
        W (np.array): adjacency matrix
        K (int): K-hop neighborhood to compute the neighbors
        N (int or 'all'): how many nodes (from top) to compute the neighbors
            from (default: 'all').
        nb (int or 'all'): how many nodes to consider valid when computing the
            neighborhood (i.e. nodes beyond nb are not trimmed out of the
            neighborhood; note that nodes smaller than nb that can be reached
            by nodes greater than nb, are included. default: 'all')
        outputType ('list' or 'matrix'): choose if the output is given in the
            form of a list of arrays, or a matrix with zero-padding of neighbors
            with neighborhoods smaller than the maximum neighborhood
            (default: 'list')

    Output:
        neighborhood (np.array or list): contains the indices of the neighboring
            nodes following the order established by the adjacency matrix.
    """
    # outputType is either a list (a list of np.arrays) or a matrix.
    assert outputType == 'list' or outputType == 'matrix'
    # Here, we can assume S is already sparse, in which case is a list of
    # sparse matrices, or that S is full, in which case it is a 3-D array.
    if isinstance(S, list):
        # If it is a list, it has to be a list of matrices, where the length
        # of the list has to be the number of edge weights. But we actually need
        # to sum over all edges to be sure we consider all reachable nodes on
        # at least one of the edge dimensions
        newS = 0.
        for e in len(S):
            # First check it's a matrix, and a square one
            assert len(S[e]) == 2
            assert S[e].shape[0] == S[e].shape[1]
            # For each edge, convert to sparse (in COO because we care about
            # coordinates to find the neighborhoods)
            newS += scipy.sparse.coo_matrix(
                              (np.abs(S[e]) > zeroTolerance).astype(S[e].dtype))
        S = (newS > zeroTolerance).astype(newS.dtype)
    else:
        # if S is not a list, check that it is either a E x N x N or a N x N
        # array.
        assert len(S.shape) == 2 or len(S.shape) == 3
        if len(S.shape) == 3:
            assert S.shape[1] == S.shape[2]
            # If it has an edge feature dimension, just add over that dimension.
            # We only need one non-zero value along the vector to have an edge
            # there. (Obs.: While normally assume that all weights are positive,
            # let's just add on abs() value to avoid any cancellations).
            S = np.sum(np.abs(S), axis = 0)
            S = scipy.sparse.coo_matrix((S > zeroTolerance).astype(S.dtype))
        else:
            # In this case, if it is a 2-D array, we do not need to add over the
            # edge dimension, so we just sparsify it
            assert S.shape[0] == S.shape[1]
            S = scipy.sparse.coo_matrix((S > zeroTolerance).astype(S.dtype))
    # Now, we finally have a sparse, binary matrix, with the connections.
    # Now check that K and N are correct inputs.
    # K is an int (target K-hop neighborhood)
    # N is either 'all' or an int determining how many rows
    assert K >= 0 # K = 0 is just the identity
    # Check how many nodes we want to obtain
    if N == 'all':
        N = S.shape[0]
    if nb == 'all':
        nb = S.shape[0]
    assert N >= 0 and N <= S.shape[0] # Cannot return more nodes than there are
    assert nb >= 0 and nb <= S.shape[0]

    # All nodes are in their own neighborhood, so
    allNeighbors = [ [n] for n in range(S.shape[0])]
    # Now, if K = 0, then these are all the neighborhoods we need.
    # And also keep track only about the nodes we care about
    neighbors = [ [n] for n in range(N)]
    # But if K > 0
    if K > 0:
        # Let's start with the one-hop neighborhood of all nodes (we need this)
        nonzeroS = list(S.nonzero())
        # This is a tuple with two arrays, the first one containing the row
        # index of the nonzero elements, and the second one containing the
        # column index of the nonzero elements.
        # Now, we want the one-hop neighborhood of all nodes (and all nodes have
        # a one-hop neighborhood, since the graphs are connected)
        for n in range(len(nonzeroS[0])):
            # The list in index 0 is the nodes, the list in index 1 is the
            # corresponding neighbor
            allNeighbors[nonzeroS[0][n]].append(nonzeroS[1][n])
        # Now that we have the one-hop neighbors, we just need to do a depth
        # first search looking for the one-hop neighborhood of each neighbor
        # and so on.
        oneHopNeighbors = allNeighbors.copy()
        # We have already visited the nodes themselves, since we already
        # gathered the one-hop neighbors.
        visitedNodes = [ [n] for n in range(N)]
        # Keep only the one-hop neighborhood of the ones we're interested in
        neighbors = [list(set(allNeighbors[n])) for n in range(N)]
        # For each hop
        for k in range(1,K):
            # For each of the nodes we care about
            for i in range(N):
                # Store the new neighbors to be included for node i
                newNeighbors = []
                # Take each of the neighbors we already have
                for j in neighbors[i]:
                    # and if we haven't visited those neighbors yet
                    if j not in visitedNodes[i]:
                        # Just look for our neighbor's one-hop neighbors and
                        # add them to the neighborhood list
                        newNeighbors.extend(oneHopNeighbors[j])
                        # And don't forget to add the node to the visited ones
                        # (we already have its one-hope neighborhood)
                        visitedNodes[i].append(j)
                # And now that we have added all the new neighbors, we add them
                # to the old neighbors
                neighbors[i].extend(newNeighbors)
                # And get rid of those that appear more than once
                neighbors[i] = list(set(neighbors[i]))

    # Now that all nodes have been collected, get rid of those beyond nb
    for i in range(N):
        # Get the neighborhood
        thisNeighborhood = neighbors[i].copy()
        # And get rid of the excess nodes
        neighbors[i] = [j for j in thisNeighborhood if j < nb]


    if outputType == 'matrix':
        # List containing all the neighborhood sizes
        neighborhoodSizes = [len(x) for x in neighbors]
        # Obtain max number of neighbors
        maxNeighborhoodSize = max(neighborhoodSizes)
        # then we have to check each neighborhood and find if we need to add
        # more nodes (itself) to pad it so we can build a matrix
        paddedNeighbors = []
        for n in range(N):
            paddedNeighbors += [np.concatenate(
                       (neighbors[n],
                        n * np.ones(maxNeighborhoodSize - neighborhoodSizes[n]))
                                )]
        # And now that every element in the list paddedNeighbors has the same
        # length, we can make it a matrix
        neighbors = np.array(paddedNeighbors, dtype = np.int)

    return neighbors

def computeSourceNodes(A, C):
    """
    computeSourceNodes: compute source nodes for the source localization problem
    
    Input:
        A (np.array): adjacency matrix of shape N x N
        C (int): number of classes
        
    Output:
        sourceNodes (list): contains the indices of the C source nodes
        
    Uses the adjacency matrix to compute C communities by means of spectral 
    clustering, and then selects the node with largest degree within each 
    community
    """
    sourceNodes = []
    degree = np.sum(A, axis = 0) # degree of each vector
    # Compute communities
    communityClusters = SpectralClustering(n_clusters = C,
                                           affinity = 'precomputed',
                                           assign_labels = 'discretize')
    communityClusters = communityClusters.fit(A)
    communityLabels = communityClusters.labels_
    # For each community
    for c in range(C):
        communityNodes = np.nonzero(communityLabels == c)[0]
        degreeSorted = np.argsort(degree[communityNodes])
        sourceNodes = sourceNodes + [communityNodes[degreeSorted[-1]]]
    
    return sourceNodes
        
        

def isConnected(W):
    """
    isConnected: determine if a graph is connected

    Input:
        W (np.array): adjacency matrix

    Output:
        connected (bool): True if the graph is connected, False otherwise
    
    Obs.: If the graph is directed, we consider it is connected when there is
    at least one edge that would make it connected (i.e. if we drop the 
    direction of all edges, and just keep them as undirected, then the resulting
    graph would be connected).
    """
    undirected = np.allclose(W, W.T, atol = zeroTolerance)
    if not undirected:
        W = 0.5 * (W + W.T)
    L = adjacencyToLaplacian(W)
    E, V = computeGFT(L)
    e = np.diag(E) # only eigenvavlues
    # Check how many values are greater than zero:
    nComponents = np.sum(e < zeroTolerance) # Number of connected components
    if nComponents == 1:
        connected = True
    else:
        connected = False
    return connected

def sparsifyGraph(W, sparsificationType, p):
    """
    sparsifyGraph: sparsifies a given graph matrix
    
    Input:
        W (np.array): adjacency matrix
        sparsificationType ('threshold' or 'NN'): threshold or nearest-neighbor
        sparsificationParameter (float): sparsification parameter (value of the
            threshold under which edges are deleted or the number of NN to keep)
        
    Output:
        W (np.array): adjacency matrix of sparsified graph
    
    Observation:
        - If it is an undirected graph, when computing the kNN edges, the
    resulting graph might be directed. Then, the graph is converted into an
    undirected one by taking the average of incoming and outgoing edges (this
    might result in a graph where some nodes have more than kNN neighbors).
        - If it is a directed graph, remember that element (i,j) of the 
    adjacency matrix corresponds to edge (j,i). This means that each row of the
    matrix has nonzero elements on all the incoming edges. In the directed case,
    the number of nearest neighbors is with respect to the incoming edges (i.e.
    kNN incoming edges are kept).
        - If the original graph is connected, then thresholding might
    lead to a disconnected graph. If this is the case, the threshold will be
    increased in small increments until the resulting graph is connected.
    To recover the actual treshold used (higher than the one specified) do
    np.min(W[np.nonzero(W)]). In the case of kNN, if the resulting graph is
    disconnected, the parameter k is increased in 1 until the resultin graph
    is connected.
    """
    # Check input arguments
    N = W.shape[0]
    assert W.shape[1] == N
    assert sparsificationType == 'threshold' or sparsificationType == 'NN'
    
    connected = isConnected(W)
    undirected = np.allclose(W, W.T, atol = zeroTolerance)
    #   np.allclose() gives true if matrices W and W.T are the same up to
    #   atol.
    
    # Start with thresholding
    if sparsificationType == 'threshold':
        Wnew = W.copy()
        Wnew[np.abs(Wnew) < p] = 0.
        # If the original graph was connected, we need to be sure this one is
        # connected as well
        if connected:
            # Check if the new graph is connected
            newGraphIsConnected = isConnected(Wnew)
            # While it is not connected
            while not newGraphIsConnected:
                # We need to reduce the size of p until we get it connected
                p = p/2.
                Wnew = W.copy()
                Wnew[np.abs(Wnew) < p] = 0.
                # Check if it is connected now
                newGraphIsConnected = isConnected(Wnew)
    # Now, let's move to k nearest neighbors
    elif sparsificationType == 'NN':
        # We sort the values of each row (in increasing order)
        Wsorted = np.sort(W, axis = 1)
        # Pick the kth largest
        kthLargest = Wsorted[:, -p] # array of size N
        # Find the elements that are greater or equal that these values
        maskOfEdgesToKeep = (W >= kthLargest.reshape([N,1])).astype(W.dtype)
        # And keep those edges
        Wnew = W * maskOfEdgesToKeep
        # If the original graph was connected
        if connected:
            # Check if the new graph is connected
            newGraphIsConnected = isConnected(Wnew)
            # While it is not connected
            while not newGraphIsConnected:
                # Increase the number of k-NN by 1
                p = p + 1
                # Compute the new kth Largest
                kthLargest = Wsorted[:, -p] # array of size N
                # Find the elements that are greater or equal that these values
                maskOfEdgesToKeep = (W >= kthLargest.reshape([N,1]))\
                                                                .astype(W.dtype)
                # And keep those edges
                Wnew = W * maskOfEdgesToKeep
                # Check if it is connected now
                newGraphIsConnected = isConnected(Wnew)
        # if it's undirected, this is the moment to reconvert it as undirected
        if undirected:
            Wnew = 0.5 * (Wnew + Wnew.T)
            
    return Wnew

def createGraph(graphType, N, graphOptions):
    """
    createGraph: creates a graph of a specified type
    
    Input:
        graphType (string): 'SBM', 'SmallWorld', 'fuseEdges', and 'adjacency'
        N (int): Number of nodes
        graphOptions (dict): Depends on the type selected.
        Obs.: More types to come.
        
    Output:
        W (np.array): adjacency matrix of shape N x N
    
    Optional inputs (by keyword):
        graphType: 'SBM'
            'nCommunities': (int) number of communities
            'probIntra': (float) probability of drawing an edge between nodes
                inside the same community
            'probInter': (float) probability of drawing an edge between nodes
                of different communities
            Obs.: This always results in a connected graph.
        graphType: 'SmallWorld'
            'probEdge': probability of drawing an edge between nodes
            'probRewiring': probability of rewiring an edge
            Obs.: This always results in a connected graph.
        graphType: 'fuseEdges'
            (Given a collection of adjacency matrices of graphs with the same
            number of nodes, this graph type is a fusion of the edges of the 
            collection of graphs, following different desirable properties)
            'adjacencyMatrices' (np.array): collection of matrices in a tensor
                np.array of dimension nGraphs x N x N
            'aggregationType' ('sum' or 'avg'): if 'sum', edges are summed
                across the collection of matrices, if 'avg' they are averaged
            'normalizationType' ('rows', 'cols' or 'no'): if 'rows', the values
                of the rows (after aggregated) are normalized to sum to one, if
                'cols', it is for the columns, if it is 'no' there is no 
                normalization.
            'isolatedNodes' (bool): if True, keep isolated nodes should there
                be any
            'forceUndirected' (bool): if True, make the resulting graph 
                undirected by replacing directed edges by the average of the 
                outgoing and incoming edges between each pair of nodes
            'forceConnected' (bool): if True, make the graph connected by taking
                the largest connected component
            'nodeList' (list): this is an empty list that, after calling the
                function, will contain a list of the nodes that were kept when
                creating the adjacency matrix out of fusing the given ones with
                the desired options
            'extraComponents' (list, optional): if the resulting fused adjacency
                matrix is not connected, and then forceConnected = True, then
                this list will contain two lists, the first one with the 
                adjacency matrices of the smaller connected components, and
                the second one a corresponding list with the index of the nodes
                that were kept for each of the smaller connected components
            (Obs.: If a given single graph is required to be adapted with any
            of the options in this function, then it can just be expanded to
            have one dimension along axis = 0 and fed to this function to
            obtain the corresponding graph with the desired properties)
        graphType: 'adjacency'
            'adjacencyMatrix' (np.array): just return the given adjacency
                matrix (after checking it has N nodes)
    """
    # Check
    assert N >= 0

    if graphType == 'SBM':
        assert(len(graphOptions.keys())) == 3
        C = graphOptions['nCommunities'] # Number of communities
        assert int(C) == C # Check that the number of communities is an integer
        pii = graphOptions['probIntra'] # Intracommunity probability
        pij = graphOptions['probInter'] # Intercommunity probability
        assert 0 <= pii <= 1 # Check that they are valid probabilities
        assert 0 <= pij <= 1
        # We create the SBM as follows: we generate random numbers between
        # 0 and 1 and then we compare them elementwise to a matrix of the
        # same size of pii and pij to set some of them to one and other to
        # zero.
        # Let's start by creating the matrix of pii and pij.
        # First, we need to know how many numbers on each community.
        nNodesC = [N//C] * C # Number of nodes per community: floor division
        c = 0 # counter for community
        while sum(nNodesC) < N: # If there are still nodes to put in communities
        # do it one for each (balanced communities)
            nNodesC[c] = nNodesC[c] + 1
            c += 1
        # So now, the list nNodesC has how many nodes are on each community.
        # We proceed to build the probability matrix.
        # We create a zero matrix
        probMatrix = np.zeros([N,N])
        # And fill ones on the block diagonals following the number of nodes.
        # For this, we need the cumulative sum of the number of nodes
        nNodesCIndex = [0] + np.cumsum(nNodesC).tolist()
        # The zero is added because it is the first index
        for c in range(C):
            probMatrix[ nNodesCIndex[c] : nNodesCIndex[c+1] , \
                        nNodesCIndex[c] : nNodesCIndex[c+1] ] = \
                np.ones([nNodesC[c], nNodesC[c]])
        # The matrix probMatrix has one in the block diagonal, which should
        # have probabilities p_ii and 0 in the offdiagonal that should have
        # probabilities p_ij. So that
        probMatrix = pii * probMatrix + pij * (1 - probMatrix)
        # has pii in the intracommunity blocks and pij in the intercommunity
        # blocks.
        # Now we're finally ready to generate a connected graph
        connectedGraph = False
        while not connectedGraph:
            # Generate random matrix
            W = np.random.rand(N,N)
            W = (W < probMatrix).astype(np.float64)
            # This matrix will have a 1 if the element ij is less or equal than
            # p_ij, so that if p_ij = 0.8, then it will be 1 80% of the times
            # (on average).
            # We need to make it undirected and without self-loops, so keep the
            # upper triangular part after the main diagonal
            W = np.triu(W, 1)
            # And add it to the lower triangular part
            W = W + W.T
            # Now let's check that it is connected
            connectedGraph = isConnected(W)
    elif graphType == 'SmallWorld':
        # Function provided by Tuomo Mäki-Marttunen
        # Connectedness introduced by Dr. S. Segarra.
        # Adapted to numpy by Fernando Gama.
        p = graphOptions['probEdge'] # Edge probability
        q = graphOptions['probRewiring'] # Rewiring probability
        # Positions on a circle
        posX = np.cos(2*np.pi*np.arange(0,N)/N).reshape([N,1]) # x axis
        posY = np.sin(2*np.pi*np.arange(0,N)/N).reshape([N,1]) # y axis
        pos = np.concatenate((posX, posY), axis = 1) # N x 2 position matrix
        connectedGraph = False
        W = np.zeros([N,N], dtype = pos.dtype) # Empty adjacency matrix
        D = sp.distance.squareform(sp.distance.pdist(pos)) ** 2 # Squared
            # distance matrix
        
        while not connectedGraph:
            # 1. The generation of locally connected network with given
            # in-degree:
            for n in range(N): # Go through all nodes in order
                nn = np.random.binomial(N, p)
                # Possible inputs are all but the node itself:
                pind = np.concatenate((np.arange(0,n), np.arange(n+1, N)))
                sortedIndices = np.argsort(D[n,pind])
                dists = D[n,pind[sortedIndices]]
                inds_equallyfar = np.nonzero(dists == dists[nn])[0]
                if len(inds_equallyfar) == 1: # if a unique farthest node to
                        # be chosen as input
                    W[pind[sortedIndices[0:nn]],n] = 1 # choose as inputs all
                        # from closest to the farthest-to-be-chosen
                else:
                    W[pind[sortedIndices[0:np.min(inds_equallyfar)]],n] = 1
                        # choose each nearer than farthest-to-be-chosen
                    r=np.random.permutation(len(inds_equallyfar)).astype(np.int)
                        # choose randomly between the ones that are as far as 
                        # be-chosen
                        
                    W[pind[sortedIndices[np.min(inds_equallyfar)\
                                    +r[0:nn-np.min(inds_equallyfar)+1]]],n] = 1;
            # 2. Watts-Strogatz perturbation:
            for n in range(N):
                A = np.nonzero(W[:,n])[0] # find the in-neighbours of n
                for j in range(len(A)):
                    if np.random.rand() < q:
                        freeind = 1 - W[:,n] # possible new candidates are
                            # all the ones not yet outputting to n
                            # (excluding n itself)
                        freeind[n] = 0
                        freeind[A[j]] = 1
                        B = np.nonzero(freeind)[0]
                        r = np.floor(np.random.rand()*len(B)).astype(np.int)
                        W[A[j],n] = 0
                        W[B[r],n] = 1;
            
            # symmetrize M
            W = np.triu(W)
            W = W + W.T
            # Check that graph is connected
            connectedGraph = isConnected(W)
    elif graphType == 'fuseEdges':
        # This alternative assumes that there are multiple graphs that have to
        # be fused into one.
        # This will be done in two ways: average or sum.
        # On top, options will include: to symmetrize it or not, to make it
        # connected or not.
        # The input data is a tensor E x N x N where E are the multiple edge
        # features that we want to fuse.
        # Argument N is ignored
        # Data
        assert 7 <= len(graphOptions.keys()) <= 8
        W = graphOptions['adjacencyMatrices'] # Data in format E x N x N
        assert len(W.shape) == 3
        N = W.shape[1] # Number of nodes
        assert W.shape[1] == W.shape[2]
        # Name the list with all nodes to keep
        nodeList = graphOptions['nodeList'] # This should be an empty list
        # If there is an 8th argument, this is where we are going to save the
        # extra components which are not the largest
        if len(graphOptions.keys()) == 8:
            logExtraComponents = True
            extraComponents = graphOptions['extraComponents']
            # This will be a list with two elements, the first elements will be
            # the adjacency matrix of the other (smaller) components, whereas
            # the second elements will be a list of the same size, where each
            # elements is yet another list of nodes to keep from the original 
            # graph to build such an adjacency matrix (akin to nodeList)
        else:
            logExtraComponents = False # Flag to know if we need to log the
            # extra components or not
        allNodes = np.arange(N)
        # What type of node aggregation
        aggregationType = graphOptions['aggregationType']
        assert aggregationType == 'sum' or aggregationType == 'avg'
        if aggregationType == 'sum':
            W = np.sum(W, axis = 0)
        elif aggregationType == 'avg':
            W = np.mean(W, axis = 0)
        # Normalization (sum of rows or columns is equal to 1)
        normalizationType = graphOptions['normalizationType']
        if normalizationType == 'rows':
            rowSum = np.sum(W, axis = 1).reshape([N, 1])
            rowSum[np.abs(rowSum) < zeroTolerance] = 1.
            W = W/np.tile(rowSum, [1, N])
        elif normalizationType == 'cols':
            colSum = np.sum(W, axis = 0).reshape([1, N])
            colSum[np.abs(colSum) < zeroTolerance] = 1.
            W = W/np.tile(colSum, [N, 1])
        # Discarding isolated nodes
        isolatedNodes = graphOptions['isolatedNodes'] # if True, isolated nodes
            # are allowed, if not, discard them
        if isolatedNodes == False:
            # A Node is isolated when it's degree is zero
            degVector = np.sum(np.abs(W), axis = 0)
            # Keep nodes whose degree is not zero
            keepNodes = np.nonzero(degVector > zeroTolerance)
            # Get the first element of the output tuple, for some reason if
            # we take keepNodes, _ as the output it says it cannot unpack it.
            keepNodes = keepNodes[0]
            if len(keepNodes) < N:
                W = W[keepNodes][:, keepNodes]
                # Update the nodes kept
                allNodes = allNodes[keepNodes]
        # Check if we need to make it undirected or not
        forceUndirected = graphOptions['forceUndirected'] # if True, make it
            # undirected by using the average between nodes (careful, some 
            # edges might cancel)
        if forceUndirected == True:
            W = 0.5 * (W + W.T)
        # Finally, making it a connected graph
        forceConnected = graphOptions['forceConnected'] # if True, make the
            # graph connected
        if forceConnected == True:
            # Check if the given graph is already connected
            connectedFlag = isConnected(W)
            # If it is not connected
            if not connectedFlag:
                # Find all connected components
                nComponents, nodeLabels = \
                                    scipy.sparse.csgraph.connected_components(W)          
                # Now, we have to pick the connected component with the largest
                # number of nodes, because that's the one to output.
                # Momentarily store the rest.
                # Let's get the list of nodes we have so far
                partialNodes = np.arange(W.shape[0])
                # Create the lists to store the adjacency matrices and
                # the official lists of nodes to keep
                eachAdjacency = [None] * nComponents
                eachNodeList = [None] * nComponents
                # And we want to keep the one with largest number of nodes, but
                # we will do only one for, so we need to be checking which one
                # is, so we will compare against the maximum number of nodes
                # registered so far
                nNodesMax = 0 # To start
                for l in range(nComponents):
                    # Find the nodes belonging to the lth connected component
                    thisNodesToKeep = partialNodes[nodeLabels == l]
                    # This adjacency matrix
                    eachAdjacency[l] = W[thisNodesToKeep][:, thisNodesToKeep]
                    # The actual list
                    eachNodeList[l] = allNodes[thisNodesToKeep]
                    # Check the number of nodes
                    thisNumberOfNodes = len(thisNodesToKeep)
                    # And see if this is the largest
                    if thisNumberOfNodes > nNodesMax:
                        # Store the new number of maximum nodes
                        nNodesMax = thisNumberOfNodes
                        # Store the element of the list that satisfies it
                        indexLargestComponent = l
                # Once we have been over all the connected components, just
                # output the one with largest number of nodes
                W = eachAdjacency.pop(indexLargestComponent)
                allNodes = eachNodeList.pop(indexLargestComponent)
                # Check that it is effectively connected
                assert isConnected(W)
                # And, if we have the extra argument, return all the other
                # connected components
                if logExtraComponents == True:
                    extraComponents.append(eachAdjacency)
                    extraComponents.append(eachNodeList)
        # To end, update the node list, so that it is returned through argument
        nodeList.extend(allNodes.tolist())
    elif graphType == 'adjacency':
        assert 'adjacencyMatrix' in graphOptions.keys()
        W = graphOptions['adjacencyMatrix']
        assert W.shape[0] == W.shape[1] == N
            
    return W

# Permutation functions

def permIdentity(S):
    """
    permIdentity: determines the identity permnutation

    Input:
        S (np.array): matrix

    Output:
        permS (np.array): matrix permuted (since, there's no permutation, it's
              the same input matrix)
        order (list): list of indices to make S become permS.
    """
    assert len(S.shape) == 2 or len(S.shape) == 3
    if len(S.shape) == 2:
        assert S.shape[0] == S.shape[1]
        S = S.reshape([1, S.shape[0], S.shape[1]])
        scalarWeights = True
    else:
        assert S.shape[1] == S.shape[2]
        scalarWeights = False
    # Number of nodes
    N = S.shape[1]
    # Identity order
    order = np.arange(N)
    # If the original GSO assumed scalar weights, get rid of the extra dimension
    if scalarWeights:
        S = S.reshape([N, N])

    return S, order.tolist()

def permDegree(S):
    """
    permDegree: determines the permutation by degree (nodes ordered from highest
        degree to lowest)

    Input:
        S (np.array): matrix

    Output:
        permS (np.array): matrix permuted
        order (list): list of indices to permute S to turn into permS.
    """
    assert len(S.shape) == 2 or len(S.shape) == 3
    if len(S.shape) == 2:
        assert S.shape[0] == S.shape[1]
        S = S.reshape([1, S.shape[0], S.shape[1]])
        scalarWeights = True
    else:
        assert S.shape[1] == S.shape[2]
        scalarWeights = False
    # Compute the degree
    d = np.sum(np.sum(S, axis = 1), axis = 0)
    # Sort ascending order (from min degree to max degree)
    order = np.argsort(d)
    # Reverse sorting
    order = np.flip(order,0)
    # And update S
    S = S[:,order,:][:,:,order]
    # If the original GSO assumed scalar weights, get rid of the extra dimension
    if scalarWeights:
        S = S.reshape([S.shape[1], S.shape[2]])

    return S, order.tolist()

def permSpectralProxies(S):
    """
    permSpectralProxies: determines the permutation by the spectral proxies
        score (from highest to lowest)

    Input:
        S (np.array): matrix

    Output:
        permS (np.array): matrix permuted
        order (list): list of indices to permute S to turn into permS.
    """
    # Design decisions:
    k = 8 # Parameter of the spectral proxies method. This is fixed for
    # consistency with the calls of the other permutation functions.
    # Design decisions: If we are given a multi-edge GSO, we're just going to
    # average all the edge dimensions and use that to compute the spectral
    # proxies.
    # Check S is of correct shape
    assert len(S.shape) == 2 or len(S.shape) == 3
    # If it is a matrix, just use it
    if len(S.shape) == 2:
        assert S.shape[0] == S.shape[1]
        scalarWeights = True
        simpleS = S.copy()
    # If it is a tensor of shape E x N x N, average over dimension E.
    else:
        assert S.shape[1] == S.shape[2]
        scalarWeights = False
        # Average over dimension E
        simpleS = np.mean(S, axis = 0)

    N = simpleS.shape[0] # Number of nodes
    ST = simpleS.conj().T # Transpose of S, needed for the method
    Sk = np.linalg.matrix_power(simpleS,k) # S^k
    STk = np.linalg.matrix_power(ST,k) # (S^T)^k
    STkSk = STk @ Sk # (S^T)^k * S^k, needed for the method

    nodes = [] # Where to save the nodes, order according the criteria
    it = 1
    M = N # This opens up the door if we want to use this code for the actual
    # selection of nodes, instead of just ordering

    while len(nodes) < M:
        remainingNodes = [n for n in range(N) if n not in nodes]
        # Computes the eigenvalue decomposition
        phi_eig, phi_ast_k = np.linalg.eig(
                STkSk[remainingNodes][:,remainingNodes])
        phi_ast_k = phi_ast_k[:][:,np.argmin(phi_eig.real)]
        abs_phi_ast_k_2 = np.square(np.absolute(phi_ast_k))
        newNodePos = np.argmax(abs_phi_ast_k_2)
        nodes.append(remainingNodes[newNodePos])
        it += 1

    if scalarWeights:
        S = S[nodes,:][:,nodes]
    else:
        S = S[:,nodes,:][:,:,nodes]
    return S, nodes

def permEDS(S):
    """
    permEDS: determines the permutation by the experimentally designed sampling
        score (from highest to lowest)

    Input:
        S (np.array): matrix

    Output:
        permS (np.array): matrix permuted
        order (list): list of indices to permute S to turn into permS.
    """
    # Design decisions: If we are given a multi-edge GSO, we're just going to
    # average all the edge dimensions and use that to compute the spectral
    # proxies.
    # Check S is of correct shape
    assert len(S.shape) == 2 or len(S.shape) == 3
    # If it is a matrix, just use it
    if len(S.shape) == 2:
        assert S.shape[0] == S.shape[1]
        scalarWeights = True
        simpleS = S.copy()
    # If it is a tensor of shape E x N x N, average over dimension E.
    else:
        assert S.shape[1] == S.shape[2]
        scalarWeights = False
        # Average over dimension E
        simpleS = np.mean(S, axis = 0)

    E, V = np.linalg.eig(simpleS) # Eigendecomposition of S
    kappa = np.max(np.absolute(V), axis=1)

    kappa2 = np.square(kappa) # The probabilities assigned to each node are
    # proportional to kappa2, so in the mean, the ones with largest kappa^2
    # would be "sampled" more often, and as suche are more important (i.e.
    # they have a higher score)

    # Sort ascending order (from min degree to max degree)
    order = np.argsort(kappa2)
    # Reverse sorting
    order = np.flip(order,0)

    if scalarWeights:
        S = S[order,:][:,order]
    else:
        S = S[:,order,:][:,:,order]

    return S, order.tolist()

def edgeFailSampling(W, p):
    """
    edgeFailSampling: randomly delete the edges of a given graph
    
    Input:
        W (np.array): adjacency matrix
        p (float): probability of deleting an edge
    
    Output:
        W (np.array): adjacency matrix with some edges randomly deleted
        
    Obs.: The resulting graph need not be connected (even if the input graph is)
    """
    
    assert 0 <= p <= 1
    N = W.shape[0]
    assert W.shape[1] == N
    undirected = np.allclose(W, W.T, atol = zeroTolerance)
    
    maskEdges = np.random.rand(N, N)
    maskEdges = (maskEdges > p).astype(W.dtype) # Put a 1 with probability 1-p
    
    W = maskEdges * W
    if undirected:
        W = np.triu(W)
        W = W + W.T
        
    return W


class Graph():
    """
    Graph: class to handle a graph with several of its properties

    Initialization:

        graphType (string): 'SBM', 'SmallWorld', 'fuseEdges', and 'adjacency'
        N (int): number of nodes
        [optionalArguments]: related to the specific type of graph; see
            createGraph() for details.

    Attributes:

        .N (int): number of nodes
        .M (int): number of edges
        .W (np.array): weighted adjacency matrix
        .D (np.array): degree matrix
        .A (np.array): unweighted adjacency matrix
        .L (np.array): Laplacian matrix (if graph is undirected and has no
           self-loops)
        .S (np.array): graph shift operator (weighted adjacency matrix by
           default)
        .E (np.array): eigenvalue (diag) matrix (graph frequency coefficients)
        .V (np.array): eigenvector matrix (graph frequency basis)
        .undirected (bool): True if the graph is undirected
        .selfLoops (bool): True if the graph has self-loops

    Methods:
    
        .computeGFT(): computes the GFT of the existing stored GSO and stores
            it internally in self.V and self.E (if this is never called, the
            corresponding attributes are set to None)

        .setGSO(S, GFT = 'no'): sets a new GSO
        Inputs:
            S (np.array): new GSO matrix (has to have the same number of nodes),
                updates attribute .S
            GFT ('no', 'increasing' or 'totalVariation'): order of
                eigendecomposition; if 'no', no eigendecomposition is made, and
                the attributes .V and .E are set to None
    """
    # in this class we provide, easily as attributes, the basic notions of
    # a graph. This serve as a building block for more complex notions as well.
    def __init__(self, graphType, N, graphOptions):
        assert N > 0
        #\\\ Create the graph (Outputs adjacency matrix):
        self.W = createGraph(graphType, N, graphOptions)
        # TODO: Let's start easy: make it just an N x N matrix. We'll see later
        # the rest of the things just as handling multiple features and stuff.
        #\\\ Number of nodes:
        self.N = (self.W).shape[0]
        #\\\ Bool for graph being undirected:
        self.undirected = np.allclose(self.W, (self.W).T, atol = zeroTolerance)
        #   np.allclose() gives true if matrices W and W.T are the same up to
        #   atol.
        #\\\ Bool for graph having self-loops:
        self.selfLoops = True \
                        if np.sum(np.abs(np.diag(self.W)) > zeroTolerance) > 0 \
                        else False
        #\\\ Degree matrix:
        self.D = np.diag(np.sum(self.W, axis = 1))
        #\\\ Number of edges:
        self.M = int(np.sum(np.triu(self.W)) if self.undirected \
                                                    else np.sum(self.W))
        #\\\ Unweighted adjacency:
        self.A = (np.abs(self.W) > 0).astype(self.W.dtype)
        #\\\ Laplacian matrix:
        #   Only if the graph is undirected and has no self-loops
        if self.undirected and not self.selfLoops:
            self.L = adjacencyToLaplacian(self.W)
        else:
            self.L = None
        #\\\ GSO (Graph Shift Operator):
        #   The weighted adjacency matrix by default
        self.S = self.W
        #\\\ GFT: Declare variables but do not compute it unless specifically
        # requested
        self.E = None # Eigenvalues
        self.V = None # Eigenvectors
    
    def computeGFT(self):
        # Compute the GFT of the stored GSO
        if self.S is not None:
            #\\ GFT:
            #   Compute the eigenvalues (E) and eigenvectors (V)
            self.E, self.V = computeGFT(self.S, order = 'totalVariation')

    def setGSO(self, S, GFT = 'no'):
        # This simply sets a matrix as a new GSO. It has to have the same number
        # of nodes (otherwise, it's a different graph!) and it can or cannot
        # compute the GFT, depending on the options for GFT
        assert S.shape[0] == S.shape[1] == self.N
        assert GFT == 'no' or GFT == 'increasing' or GFT == 'totalVariation'
        # Set the new GSO
        self.S = S
        if GFT == 'no':
            self.E = None
            self.V = None
        else:
            self.E, self.V = computeGFT(self.S, order = GFT)

def splineBasis(K, x, degree=3):
    # Function written by M. Defferrard, taken verbatim (except for function
    # name), from 
    # https://github.com/mdeff/cnn_graph/blob/master/lib/models.py#L662
    """
    Return the B-spline basis.
    K: number of control points.
    x: evaluation points
       or number of evenly distributed evaluation points.
    degree: degree of the spline. Cubic spline by default.
    """
    if np.isscalar(x):
        x = np.linspace(0, 1, x)

    # Evenly distributed knot vectors.
    kv1 = x.min() * np.ones(degree)
    kv2 = np.linspace(x.min(), x.max(), K-degree+1)
    kv3 = x.max() * np.ones(degree)
    kv = np.concatenate((kv1, kv2, kv3))

    # Cox - DeBoor recursive function to compute one spline over x.
    def cox_deboor(k, d):
        # Test for end conditions, the rectangular degree zero spline.
        if (d == 0):
            return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
    basis[-1,-1] = 1
    return basis

def coarsen(A, levels, self_connections=False):
    # Function written by M. Defferrard, taken (almost) verbatim, from 
    # https://github.com/mdeff/cnn_graph/blob/master/lib/coarsening.py#L5
    """
    Coarsen a graph, represented by its adjacency matrix A, at multiple
    levels.
    """
    graphs, parents = metis(A, levels)
    perms = compute_perm(parents)

    for i, A in enumerate(graphs):
        M, M = A.shape

        if not self_connections:
            A = A.tocoo()
            A.setdiag(0)

        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr()
        A.eliminate_zeros()
        graphs[i] = A

#        Mnew, Mnew = A.shape
#        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
#              '|E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))


    return graphs, perms[0] if levels > 0 else None

def metis(W, levels, rid=None):
    # Function written by M. Defferrard, taken verbatim, from 
    # https://github.com/mdeff/cnn_graph/blob/master/lib/coarsening.py#L34
    """
    Coarsen a graph multiple times using the METIS algorithm.
    INPUT
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs
    OUTPUT
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode
        in the graph{i}
    NOTE
        if "graph" is a list of length k, then "parents" will be a list of
        length k-1
    """

    N, N = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)
    #supernode_size = np.ones(N)
    #nd_sz = [supernode_size]
    #count = 0

    #while N > maxsize:
    for _ in range(levels):

        #count += 1

        # CHOOSE THE WEIGHTS FOR THE PAIRING
        # weights = ones(N,1)       # metis weights
        weights = degree            # graclus weights
        # weights = supernode_size  # other possibility
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = metis_one_level(rr,cc,vv,rid,weights)  # rr is ordered
        parents.append(cluster_id)

        # TO DO
        # COMPUTE THE SIZE OF THE SUPERNODES AND THEIR DEGREE 
        #supernode_size = full(   sparse(cluster_id,  ones(N,1) ,
        #	supernode_size )     )
        #print(cluster_id)
        #print(supernode_size)
        #nd_sz{count+1}=supernode_size;

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = scipy.sparse.csr_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
        W.eliminate_zeros()
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        degree = W.sum(axis=0)
        #degree = W.sum(axis=0) - W.diagonal()

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        #[~, rid]=sort(ss);     # arthur strategy
        #[~, rid]=sort(supernode_size);    #  thomas strategy
        #rid=randperm(N);                  #  metis/graclus strategy
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents

# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def metis_one_level(rr,cc,vv,rid,weights):
    # Function written by M. Defferrard, taken verbatim, from 
    # https://github.com/mdeff/cnn_graph/blob/master/lib/coarsening.py#L119

    nnz = rr.shape[0]
    N = rr[nnz-1] + 1

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id

def compute_perm(parents):
    # Function written by M. Defferrard, taken verbatim, from 
    # https://github.com/mdeff/cnn_graph/blob/master/lib/coarsening.py#L167
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    """

    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:
        #print('parent: {}'.format(parent))

        # Fake nodes go after real ones.
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2
            #print('indices_node: {}'.format(indices_node))

            # Add a node to go with a singelton.
            if len(indices_node) == 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
                #print('new singelton: {}'.format(indices_node))
            # Add two nodes as children of a singelton in the parent.
            elif len(indices_node) == 0:
                indices_node.append(pool_singeltons+0)
                indices_node.append(pool_singeltons+1)
                pool_singeltons += 2
                #print('singelton childrens: {}'.format(indices_node))

            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    # Sanity checks.
    for i,indices_layer in enumerate(indices):
        M = M_last*2**i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices[0] == M)
        # The new ordering does not omit an indice.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]

def perm_adjacency(A, indices):
    # Function written by M. Defferrard, taken verbatim, from 
    # https://github.com/mdeff/cnn_graph/blob/master/lib/coarsening.py#L242
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()

    # Add Mnew - M isolated vertices.
    if Mnew > M:
        rows = scipy.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
        cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])

    # Permute the rows and the columns.
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    # assert np.abs(A - A.T).mean() < 1e-9
    assert type(A) is scipy.sparse.coo.coo_matrix
    return A

def permCoarsening(x, indices):
    # Original function written by M. Defferrard, found in
    # https://github.com/mdeff/cnn_graph/blob/master/lib/coarsening.py#L219
    # Function name has been changed, and it has been further adapted to handle
    # multiple features as
    #   number_data_points x number_features x number_nodes
    # instead of the original
    #   number_data_points x number_nodes
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    B, F, N = x.shape
    Nnew = len(indices)
    assert Nnew >= N
    xnew = np.empty((B, F, Nnew))
    for i,j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < N:
            xnew[:,:,i] = x[:,:,j]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[:,:,i] = np.zeros([B, F])
    return xnew
