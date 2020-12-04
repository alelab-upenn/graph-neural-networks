# 2019/12/31~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
# Kate Tolstaya, eig@seas.upenn.edu
"""
architecturesTime.py Architectures module

Definition of GNN architectures. The basic idea of these architectures is that
the data comes in the form {(S_t, x_t)} where the shift operator as well as the
signal change with time, and where each training point consists of a trajectory.
Unlike architectures.py where the shift operator S is fixed (although it can
be changed after the architectures has been initialized) and the training set
consist of a set of {x_b} with b=1,...,B for a total of B samples, here the
training set is assumed to be a trajectory, and to include a different shift
operator for each sample {(S_t, x_t)_{t=1}^{T}}_{b=1,...,B}. Also, all 
implementations consider a unit delay exchange (i.e. the S_t and x_t values
get delayed by one unit of time for each neighboring exchange).

LocalGNN_DB: implements the selection GNN architecture by means of local
    operations only
GraphRecurrentNN_DB: implements the GRNN architecture
AggregationGNN_DB: implements the aggregation GNN architecture
"""

import numpy as np
import torch
import torch.nn as nn

import alegnn.utils.graphML as gml

zeroTolerance = 1e-9 # Absolute values below this number are considered zero.

class LocalGNN_DB(nn.Module):
    """
    LocalGNN_DB: implement the local GNN architecture where all operations are
        implemented locally, i.e. by means of neighboring exchanges only. More
        specifically, it has graph convolutional layers, but the readout layer,
        instead of being an MLP for the entire graph signal, it is a linear
        combination of the features at each node. It considers signals
        that change in time with batch GSOs.

    Initialization:

        LocalGNN_DB(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                    nonlinearity, # Nonlinearity
                    dimReadout, # Local readout layer
                    dimEdgeFeatures) # Structure

        Input:
            /** Graph convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nFilterTaps (list of int): number of filter taps on each layer
                (i.e. nFilterTaps-1 is the extent of neighborhoods that are
                 reached, for example K=2 is info from the 1-hop neighbors)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Readout layers **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).
                
            /** Graph structure **/
            dimEdgeFeatures (int): number of edge features

        Output:
            nn.Module with a Local GNN architecture with the above specified
            characteristics that considers time-varying batch GSO and delayed
            signals

    Forward call:

        LocalGNN_DB(x, S)

        Input:
            x (torch.tensor): input data of shape
                batchSize x timeSamples x dimFeatures x numberNodes
            GSO (torch.tensor): graph shift operator; shape
                batchSize x timeSamples (x dimEdgeFeatures)
                                                    x numberNodes x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the GNN; 
                batchSize x timeSamples x dimReadout[-1] x numberNodes
                
    Other methods:
            
        y, yGNN = .splitForward(x, S): gives the output of the entire GNN y,
        which has shape batchSize x timeSamples x dimReadout[-1] x numberNodes,
        as well as the output of all the GNN layers (i.e. before the readout
        layers), yGNN of shape batchSize x timeSamples x dimFeatures[-1]
        x numberNodes. This can be used to isolate the effect of the graph
        convolutions from the effect of the readout layer.
        
        y = .singleNodeForward(x, S, nodes): outputs the value of the last
        layer at a single node. x is the usual input of shape batchSize 
        x timeSamples x dimFeatures x numberNodes. nodes is either a single
        node (int) or a collection of nodes (list or numpy.array) of length
        batchSize, where for each element in the batch, we get the output at
        the single specified node. The output y is of shape batchSize 
        x timeSamples x dimReadout[-1].
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 dimEdgeFeatures):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = dimEdgeFeatures # Number of edge features
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.dimReadout = dimReadout
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter_DB(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            #\\ Nonlinearity
            gfl.append(self.sigma())
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0: # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph 
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.F[-1], dimReadout[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimReadout)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l+1],
                                    bias = self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.

    def splitForward(self, x, S):

        # Check the dimensions of the input
        #   S: B x T (x E) x N x N
        #   x: B x T x F[0] x N
        assert len(S.shape) == 4 or len(S.shape) == 5
        if len(S.shape) == 4:
            S = S.unsqueeze(2)
        B = S.shape[0]
        T = S.shape[1]
        assert S.shape[2] == self.E
        N = S.shape[3]
        assert S.shape[4] == N
        
        assert len(x.shape) == 4
        assert x.shape[0] == B
        assert x.shape[1] == T
        assert x.shape[2] == self.F[0]
        assert x.shape[3] == N
        
        # Add the GSO at each layer
        for l in range(self.L):
            self.GFL[2*l].addGSO(S)
        # Let's call the graph filtering layer
        yGFL = self.GFL(x)
        # Change the order, for the readout
        y = yGFL.permute(0, 1, 3, 2) # B x T x N x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x T x N x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 1, 3, 2), yGFL
        # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N
    
    def forward(self, x, S):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x, S)
        
        return output
    
    def singleNodeForward(self, x, S, nodes):
        
        # x is of shape B x T x F[0] x N
        batchSize = x.shape[0]
        N = x.shape[3]
        
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
                or type(nodes) is list \
                or type(nodes) is np.ndarray
        
        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x 1 x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x T x dimRedout[-1] x 1
        # and we just squeeze the last dimension
        
        # TODO: The big question here is if multiplying by a matrix is faster
        # than doing torch.index_select
        
        # Let's always work with numpy arrays to make it easier.
        if type(nodes) is int:
            # Change the node number to accommodate the new order
            nodes = self.order.index(nodes)
            # If it's int, make it a list and an array
            nodes = np.array([nodes], dtype=np.int)
            # And repeat for the number of batches
            nodes = np.tile(nodes, batchSize)
        if type(nodes) is list:
            newNodes = [self.order.index(n) for n in nodes]
            nodes = np.array(newNodes, dtype = np.int)
        elif type(nodes) is np.ndarray:
            newNodes = np.array([np.where(np.array(self.order) == n)[0][0] \
                                                                for n in nodes])
            nodes = newNodes.astype(np.int)
        # Now, nodes is an np.int np.ndarray with shape batchSize
        
        # Build the selection matrix
        selectionMatrix = np.zeros([batchSize, 1, N, 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype = x.dtype,
                                       device = x.device)
        
        # Now compute the output
        y = self.forward(x, S)
        # This output is of size B x T x dimReadout[-1] x N
        
        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x T x dimReadout[-1] x 1
        
        # Squeeze the last dimension and return
        return y.squeeze(3)
    
class GraphRecurrentNN_DB(nn.Module):
    """
    GraphRecurrentNN_DB: implements the GRNN architecture on a time-varying GSO
        batch and delayed signals. It is a single-layer GRNN and the hidden
        state is initialized at random drawing from a standard gaussian.
    
    Initialization:
        
        GraphRecurrentNN_DB(dimInputSignals, dimOutputSignals,
                            dimHiddenSignals, nFilterTaps, bias, # Filtering
                            nonlinearityHidden, nonlinearityOutput,
                            nonlinearityReadout, # Nonlinearities
                            dimReadout, # Local readout layer
                            dimEdgeFeatures) # Structure
        
        Input:
            /** Graph convolutions **/
            dimInputSignals (int): dimension of the input signals
            dimOutputSignals (int): dimension of the output signals
            dimHiddenSignals (int): dimension of the hidden state
            nFilterTaps (list of int): a list with two elements, the first one
                is the number of filter taps for the filters in the hidden
                state equation, the second one is the number of filter taps
                for the filters in the output
            bias (bool): include bias after graph filter on every layer
            
            /** Activation functions **/
            nonlinearityHidden (torch.function): the nonlinearity to apply
                when computing the hidden state; it has to be a torch function,
                not a nn.Module
            nonlinearityOutput (torch.function): the nonlinearity to apply when
                computing the output signal; it has to be a torch function, not
                a nn.Module.
            nonlinearityReadout (nn.Module): the nonlinearity to apply at the
                end of the readout layer (if the readout layer has more than
                one layer); this one has to be a nn.Module, instead of just a
                torch function.
                
            /** Readout layer **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).
                
            /** Graph structure **/
            dimEdgeFeatures (int): number of edge features
            
        Output:
            nn.Module with a GRNN architecture with the above specified
            characteristics that considers time-varying batch GSO and delayed
            signals
    
    Forward call:
        
        GraphRecurrentNN_DB(x, S)
        
        Input:
            x (torch.tensor): input data of shape
                batchSize x timeSamples x dimInputSignals x numberNodes
            GSO (torch.tensor): graph shift operator; shape
                batchSize x timeSamples (x dimEdgeFeatures)
                                                    x numberNodes x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the GRNN; 
                batchSize x timeSamples x dimReadout[-1] x numberNodes
        
    Other methods:
            
        y, yGNN = .splitForward(x, S): gives the output of the entire GRNN y,
        which has shape batchSize x timeSamples x dimReadout[-1] x numberNodes,
        as well as the output of the GRNN (i.e. before the readout layers), 
        yGNN of shape batchSize x timeSamples x dimInputSignals x numberNodes. 
        This can be used to isolate the effect of the graph convolutions from 
        the effect of the readout layer.
        
        y = .singleNodeForward(x, S, nodes): outputs the value of the last
        layer at a single node. x is the usual input of shape batchSize 
        x timeSamples x dimInputSignals x numberNodes. nodes is either a single
        node (int) or a collection of nodes (list or numpy.array) of length
        batchSize, where for each element in the batch, we get the output at
        the single specified node. The output y is of shape batchSize 
        x timeSamples x dimReadout[-1].
    """
    def __init__(self,
                 # Graph filtering
                 dimInputSignals,
                 dimOutputSignals,
                 dimHiddenSignals,
                 nFilterTaps, bias,
                 # Nonlinearities
                 nonlinearityHidden,
                 nonlinearityOutput,
                 nonlinearityReadout, # nn.Module
                 # Local MLP in the end
                 dimReadout,
                 # Structure
                 dimEdgeFeatures):
        # Initialize parent:
        super().__init__()
        
        # A list of two int, one for the number of filter taps (the computation
        # of the hidden state has the same number of filter taps)
        assert len(nFilterTaps) == 2
        
        # Store the values (using the notation in the paper):
        self.F = dimInputSignals # Number of input features
        self.G = dimOutputSignals # Number of output features
        self.H = dimHiddenSignals # NUmber of hidden features
        self.K = nFilterTaps # Filter taps
        self.E = dimEdgeFeatures # Number of edge features
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearityHidden
        self.rho = nonlinearityOutput
        self.nonlinearityReadout = nonlinearityReadout
        self.dimReadout = dimReadout
        #\\\ Hidden State RNN \\\
        # Create the layer that generates the hidden state, and generate z0
        self.hiddenState = gml.HiddenState_DB(self.F, self.H, self.K[0],
                                       nonlinearity = self.sigma, E = self.E,
                                       bias = self.bias)
        #\\\ Output Graph Filters \\\
        self.outputState = gml.GraphFilter_DB(self.H, self.G, self.K[1],
                                              E = self.E, bias = self.bias)
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0: # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph 
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.G, dimReadout[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimReadout)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.nonlinearityReadout())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l+1],
                                    bias = self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def splitForward(self, x, S):

        # Check the dimensions of the input
        #   S: B x T (x E) x N x N
        #   x: B x T x F[0] x N
        assert len(S.shape) == 4 or len(S.shape) == 5
        if len(S.shape) == 4:
            S = S.unsqueeze(2)
        B = S.shape[0]
        T = S.shape[1]
        assert S.shape[2] == self.E
        N = S.shape[3]
        assert S.shape[4] == N
        
        assert len(x.shape) == 4
        assert x.shape[0] == B
        assert x.shape[1] == T
        assert x.shape[2] == self.F
        assert x.shape[3] == N
        
        # This can be generated here or generated outside of here, not clear yet
        # what's the most coherent option
        z0 = torch.randn((B, self.H, N), device = x.device)
        
        # Add the GSO for each graph filter
        self.hiddenState.addGSO(S)
        self.outputState.addGSO(S)
        
        # Compute the trajectory of hidden states
        z, _ = self.hiddenState(x, z0)
        # Compute the output trajectory from the hidden states
        yOut = self.outputState(z)
        yOut = self.rho(yOut) # Don't forget the nonlinearity!
        #   B x T x G x N
        # Change the order, for the readout
        y = yOut.permute(0, 1, 3, 2) # B x T x N x G
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x T x N x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 1, 3, 2), yOut
        # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N
    
    def forward(self, x, S):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x, S)
        
        return output
        
    def singleNodeForward(self, x, S, nodes):
        
        # x is of shape B x T x F[0] x N
        batchSize = x.shape[0]
        N = x.shape[3]
        
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
                or type(nodes) is list \
                or type(nodes) is np.ndarray
        
        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x 1 x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x T x dimRedout[-1] x 1
        # and we just squeeze the last dimension
        
        # TODO: The big question here is if multiplying by a matrix is faster
        # than doing torch.index_select
        
        # Let's always work with numpy arrays to make it easier.
        if type(nodes) is int:
            # Change the node number to accommodate the new order
            nodes = self.order.index(nodes)
            # If it's int, make it a list and an array
            nodes = np.array([nodes], dtype=np.int)
            # And repeat for the number of batches
            nodes = np.tile(nodes, batchSize)
        if type(nodes) is list:
            newNodes = [self.order.index(n) for n in nodes]
            nodes = np.array(newNodes, dtype = np.int)
        elif type(nodes) is np.ndarray:
            newNodes = np.array([np.where(np.array(self.order) == n)[0][0] \
                                                                for n in nodes])
            nodes = newNodes.astype(np.int)
        # Now, nodes is an np.int np.ndarray with shape batchSize
        
        # Build the selection matrix
        selectionMatrix = np.zeros([batchSize, 1, N, 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype = x.dtype,
                                       device = x.device)
        
        # Now compute the output
        y = self.forward(x, S)
        # This output is of size B x T x dimReadout[-1] x N
        
        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x T x dimReadout[-1] x 1
        
        # Squeeze the last dimension and return
        return y.squeeze(3)

class AggregationGNN_DB(nn.Module):
    """
    AggregationGNN_DB: implement the aggregation GNN architecture with delayed
        time structure and batch GSOs

    Initialization:

        Input:
            /** Regular convolutional layers **/
            dimFeatures (list of int): number of features on each layer
            nFilterTaps (list of int): number of filter taps on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimFeatures[0] is the number of features (the dimension
                of the node signals) of the data, where dimFeatures[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimFeatures) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            poolingFunction (torch.nn): module from torch.nn pooling layers
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layer **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers after the filters have
                been applied
                
            /** Graph structure **/
            dimEdgeFeatures (int): number of edge features
            nExchanges (int): maximum number of neighborhood exchanges

        Output:
            nn.Module with an Aggregation GNN architecture with the above
            specified characteristics.

    Forward call:

        Input:
            x (torch.tensor): input data of shape
                batchSize x timeSamples x dimFeatures x numberNodes
            GSO (torch.tensor): graph shift operator of shape
                batchSize x timeSamples (x dimEdgeFeatures)
                                                     x numberNodes x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x x timeSamples x dimReadout[-1] x nNodes
    """
    def __init__(self,
                 # Graph filtering
                 dimFeatures, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 poolingFunction, poolingSize,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 dimEdgeFeatures, nExchanges):
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimFeatures) == len(nFilterTaps) + 1
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of convolutional layers
        self.F = dimFeatures # Features
        self.K = nFilterTaps # Filter taps
        self.E = dimEdgeFeatures # Dimension of edge features
        self.bias = bias # Boolean
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize # This acts as both the kernel_size and the
        # stride, so there is no overlap on the elements over which we take
        # the maximum (this is how it works as default)
        self.dimReadout = dimReadout
        self.nExchanges = nExchanges # Number of exchanges
        # Let's also record the number of nodes on each layer (L+1, actually)
        self.N = [self.nExchanges+1] # If we have one exchange, then we have
        #   two entries in the collected vector (the zeroth-exchange the
        #   first exchange)
        for l in range(self.L):
            # In pyTorch, the convolution is a valid correlation, instead of a
            # full one, which means that the output is smaller than the input.
            # Precisely, this smaller (check documentation for nn.conv1d)
            outConvN = self.N[l] - (self.K[l] - 1) # Size of the conv output
            # The next equation to compute the number of nodes is obtained from
            # the maxPool1d help in the pytorch documentation
            self.N += [int(
                           (outConvN - (self.alpha[l]-1) - 1)/self.alpha[l] + 1
                                    )]
            # int() on a float always applies floor()

        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        convl = [] # Convolutional Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            convl.append(nn.Conv1d(self.F[l]*self.E,
                                   self.F[l+1]*self.E,
                                   self.K[l],
                                   bias = self.bias))
            #\\ Nonlinearity
            convl.append(self.sigma())
            #\\ Pooling
            convl.append(self.rho(self.alpha[l]))
        # And now feed them into the sequential
        self.ConvLayers = nn.Sequential(*convl) # Convolutional layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputReadout = self.N[-1] * self.F[-1] * self.E
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputReadout,dimReadout[0],bias=self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimReadout)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l+1],
                                    bias = self.bias))
        # And we're done within each node
        self.Readout = nn.Sequential(*fc)

    def forward(self, x, S):
        
        # Check the dimensions of the input first
        #   S: B x T (x E) x N x N
        #   x: B x T x F[0] x N
        assert len(S.shape) == 4 or len(S.shape) == 5
        if len(S.shape) == 4:
            # Then S is B x T x N x N
            S = S.unsqueeze(2) # And we want it B x T x 1 x N x N
        B = S.shape[0]
        T = S.shape[1]
        assert S.shape[2] == self.E
        N = S.shape[3]
        assert S.shape[4] == N
        #   Check the dimensions of x
        assert len(x.shape) == 4
        assert x.shape[0] == B
        assert x.shape[1] == T
        assert x.shape[2] == self.F[0]
        assert x.shape[3] == N
        
        # Now we need to do the exchange to build the aggregation vector at
        # every node
        # z has to be of shape: B x T x F[0] x (nExchanges+1) x N
        # to be fed into conv1d it has to be (B*T*N) x F[0] x (nExchanges+1)
        
        # This vector is built by multiplying x with S, so we need to adapt x
        # to have a dimension that can be multiplied by S (we need to add the
        # E dimension)
        x = x.reshape([B, T, 1, self.F[0], N]).repeat(1, 1, self.E, 1, 1)
        
        # The first element of z is, precisely, this element (no exchanges)
        z = x.reshape([B, T, 1, self.E, self.F[0], N]) # The new dimension is
        #   the one that accumulates the nExchanges
        
        # Now we start with the exchanges (multiplying by S)
        for k in range(1, self.nExchanges+1):
            # Across dim = 1 (time) we need to "displace the dimension down", 
            # i.e. where it used to be t = 1 we now need it to be t=0 and so
            # on. For t=0 we add a "row" of zeros.
            x, _ = torch.split(x, [T-1, 1], dim = 1)
            #   The second part is the most recent time instant which we do 
            #   not need anymore (it's used only once for the first value of K)
            # Now, we need to add a "row" of zeros at the beginning (for t = 0)
            zeroRow = torch.zeros(B, 1, self.E, self.F[0], N, 
                                  dtype=x.dtype,device=x.device)
            x = torch.cat((zeroRow, x), dim = 1)
            # And now we multiply with S
            x = torch.matmul(x, S)
            # Add the dimension along K
            xS = x.reshape(B, T, 1, self.E, self.F[0], N)
            # And concatenate it with z
            z = torch.cat((z, xS), dim = 2)
        
        # Now, we have finally built the vector of delayed aggregations. This
        # vector has shape B x T x (nExchanges+1) x E x F[0] x N
        # To get rid of the edge features (dim E) we just sum through that
        # dimension
        z = torch.sum(z, dim = 3) # B x T x (nExchanges+1) x F[0] x N
        # It is, essentially, a matrix of N x (nExchanges+1) for each feature,
        # for each time instant, for each batch.
        # NOTE1: This is inconsequential if self.E = 1 (most of the cases)
        # NOTE2: Alternatively, not to lose information, we could contatenate
        # dim E after dim F[0] to get E*F[0] features; this increases the
        # dimensionsonality of the data (which could be fine) but need to be
        # adapted so that the first input in the conv1d takes self.E*self.F[0]
        # features instead of just self.F[0]
            
        # The operation conv1d takes tensors of shape 
        #   batchSize x nFeatures x nEntries
        # This means that the convolution takes place along nEntries with
        # a summation along nFeatures, for each of the elements along
        # batchSize. So we need to put (nExchanges+1) last since it is along
        # those elements that we want the convolution to be performed, and
        # we need to put F[0] as nFeatures since there is where we want the
        # features to be combined. The other three dimensions are different
        # elements (agents, time, batch) to which the convolution needs to be
        # applied.
        # Therefore, we want a vector z of shape
        #   (B*T*N) x F[0] x (nExchanges+1)
        
        # Let's get started with this reorganization
        #   First, we join B*T*N. Because we always join the last dimensions,
        #   we need to permute first to put B, T, N as the last dimensions.
        #   z: B x T x (nExchanges+1) x F[0] x N
        z = z.permute(3, 2, 0, 1, 4) # F[0] x (nExchanges+1) x B x T x N
        z = z.reshape([self.F[0], self.nExchanges+1, B*T*N])
        #   F[0] x (nExchanges+1) x B*T*N
        #   Second, we put it back at the beginning
        z = z.permute(2, 0, 1) # B*T*N x F[0] x (nExchanges+1)
        
        # Let's call the convolutional layers
        y = self.ConvLayers(z)
        #   B*T*N x F[-1] x N[-1]
        # Flatten the output
        y = y.reshape([B*T*N, self.F[-1] * self.N[-1]])
        # And, feed it into the per node readout layers
        y = self.Readout(y) # (B*T*N) x dimReadout[-1]
        # And now we have to unpack it back for every node, i.e. to get it
        # back to shape B x T x N x dimReadout[-1]
        y = y.permute(1, 0) # dimReadout[-1] x (B*T*N)
        y = y.reshape(self.dimReadout[-1], B, T, N)
        # And finally put it back to the usual B x T x F x N
        y = y.permute(1, 2, 0, 3)
        return y

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
