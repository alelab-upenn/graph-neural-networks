# 2021/03/04~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
architectures.py Architectures module

Definition of GNN architectures.

SelectionGNN: implements the selection GNN architecture
LocalActivationGNN: implements the selection GNN architecture with a local
    activation function (instead of pointwise)
LocalGNN: implements the selection GNN architecture by means of local
    operations only
SpectralGNN: implements the selection GNN architecture using spectral filters
NodeVariantGNN: implements the selection GNN architecture with node-variant
    graph filters
EdgeVariantGNN: implements the selection GNN architecture with edge-variant
    graph filters
LocalEdgeNet: implements the selection GNN architecture with edge-variant
    graph filters and local operations only
ARMAfilterGNN: implements the selection GNN architecture using ARMA graph
    filters by Jacobi's method
LocalARMA: implements the selection GNN architecture using ARMA graph filters
    by Jacobi's method and local operations only
AggregationGNN: implements the aggregation GNN architecture
MultiNodeAggregationGNN: implementes the multi-node aggregation GNN architecture
GraphAttentionNetwork: implement the graph attention network architecture
GraphConvolutionAttentionNetwork: implement the graph convolution attention
    network (GCAT) architecture
EdgeVariantAttention: implement the edge variant graph filter, with coefficients
    learned following a parameterization given by the attention mechanism
GraphRecurrentNN: implements a graph recurrent neural network with static GSO
GatedGraphRecurrentNN: implements a gated graph recurrent neural network with 
    static GSO. Gates can be time, node or edge gates
"""

import numpy as np
import scipy
import torch
import torch.nn as nn

import alegnn.utils.graphML as gml
import alegnn.utils.graphTools

from alegnn.utils.dataTools import changeDataType

zeroTolerance = 1e-9 # Absolute values below this number are considered zero.

class SelectionGNN(nn.Module):
    """
    SelectionGNN: implement the selection GNN architecture

    Initialization:

        SelectionGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                     nonlinearity, # Nonlinearity
                     nSelectedNodes, poolingFunction, poolingSize, # Pooling
                     dimLayersMLP, # MLP in the end
                     GSO, order = None, # Structure
                     coarsening = False)

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
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            >> Obs.: If coarsening = True, this variable is ignored since the
                number of nodes in each layer is given by the graph coarsening
                algorithm.
            poolingFunction (nn.Module in Utils.graphML or in torch.nn): 
                summarizing function
            >> Obs.: If coarsening = True, then the pooling function is one of
                the regular 1-d pooling functions available in torch.nn (instead
                of one of the summarizing functions in Utils.graphML).
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            >> Obs.: If coarsening = True, then the pooling size is ignored 
                since, due to the binary tree nature of the graph coarsening
                algorithm, it always has to be 2.
                
            /** Readout layers **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array
            coarsening (bool, default = False): if True uses graph coarsening
                instead of zero-padding to reduce the number of nodes.
            >> Obs.: (i) Graph coarsening only works when the number
                 of edge features is 1 -scalar weights-. (ii) The graph
                 coarsening forces a given order of the nodes, and this order
                 has to be used to reordering the GSO as well as the samples
                 during training; as such, this order is internally saved and
                 applied to the incoming samples in the forward call -it is
                 thus advised to use the identity ordering in the model class
                 when using the coarsening method-.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        SelectionGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                
    Other methods:
        
        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO,
                 # Ordering
                 order = None,
                 # Coarsening
                 coarsening = False):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
     
        self.coarsening = coarsening # Whether to do coarsening or not
        # If we have to do coarsening, then note that it can only be done if
        # we have a single edge feature, otherwise, each edge feature could be
        # coarsed (and thus, ordered) in a different way, and there is no
        # sensible way of merging back this different orderings. So, we will
        # only do coarsening if we have a single edge feature; otherwise, we
        # will default to selection sampling (therefore, always specify
        # nSelectedNodes)
        if self.coarsening and self.E == 1:
            self.permFunction = alegnn.utils.graphTools.permCoarsening # Override
                # permutation function for the one corresponding to coarsening
            GSO = scipy.sparse.csr_matrix(GSO[0])
            GSO, self.order = alegnn.utils.graphTools.coarsen(GSO, levels=self.L,
                                                              self_connections=False)
            # Now, GSO is a list of csr_matrix with self.L+1 coarsened GSOs,
            # we need to torch.tensor them and put them in a list.
            # order is just a list of indices to reorder the nodes.
            self.S = []
            self.N = [] # It has to be reset, because now the number of
                # nodes is determined by the coarsening scheme
            for S in GSO:
                S = S.todense().A.reshape([self.E, S.shape[0], S.shape[1]])
                    # So, S.todense() returns a numpy.matrix object; a numpy
                    # matrix cannot be converted into a tensor (i.e., added
                    # the third dimension), therefore we need to convert it to
                    # a numpy.array. According to the documentation, the 
                    # attribute .A in a numpy.matrix returns self as an ndarray
                    # object. So that's why the .A is there.
                self.S.append(torch.tensor(S))
                self.N.append(S.shape[1])
            # Finally, because the graph coarsening algorithm is a binary tree
            # pooling, we always need to force a pooling size of 2
            self.alpha = [2] * self.L
        else:
            # Call the corresponding ordering function. Recall that if no
            # order was selected, then this is permIdentity, so that nothing
            # changes.
            self.S, self.order = self.permFunction(GSO)
            if 'torch' not in repr(self.S.dtype):
                self.S = torch.tensor(self.S)
            self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
            self.alpha = poolingSize
            self.coarsening = False # If it failed because there are more than
                # one edge feature, then just set this to false, so we do not
                # need to keep checking whether self.E == 1 or not, just this
                # one
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            if self.coarsening:
                gfl[3*l].addGSO(self.S[l])
            else:
                gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma())
            #\\ Pooling
            if self.coarsening:
                gfl.append(self.rho(self.alpha[l]))
            else:
                gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
                # Same as before, this is 3*l+2
                gfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def changeGSO(self, GSO, nSelectedNodes = [], poolingSize = []):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
            
        # Now, if we don't have coarsening, then we need to reorder the GSO,
        # and since this GSO reordering will affect several parts of the non
        # coarsening algorithm, then we will do it now
        # Reorder the GSO
        if not self.coarsening:
            self.S, self.order = self.permFunction(GSO)
            # Change data type and device as required
            self.S = changeDataType(self.S, dataType)
            if device is not None:
                self.S = self.S.to(device)
            
        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0 and not self.coarsening:
            # (If it's coarsening, then the pooling size cannot change)
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize
        
        # Now, check if we have a new list of nodes (this only makes sense
        # if there is no coarsening, because if it is coarsening, the list with
        # the number of nodes to be considered is ignored.)
        if len(nSelectedNodes) > 0 and not self.coarsening:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list 
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.GFL[3*l+2] = self.rho(self.N[l], self.N[l+1],
                                           self.alpha[l])
                self.GFL[3*l+2].addGSO(self.S)
        elif len(nSelectedNodes) == 0 and not self.coarsening:
            # Just update the GSO
            for l in range(self.L):
                self.GFL[3*l+2].addGSO(self.S)
        
        # If it's coarsening, then we need to compute the new coarsening
        # scheme
        if self.coarsening and self.E == 1:
            device = self.S[0].device
            GSO = scipy.sparse.csr_matrix(GSO[0])
            GSO, self.order = alegnn.utils.graphTools.coarsen(GSO, levels=self.L,
                                                              self_connections=False)
            # Now, GSO is a list of csr_matrix with self.L+1 coarsened GSOs,
            # we need to torch.tensor them and put them in a list.
            # order is just a list of indices to reorder the nodes.
            self.S = []
            self.N = [] # It has to be reset, because now the number of
                # nodes is determined by the coarsening scheme
            for S in GSO:
                S = S.todense().A.reshape([self.E, S.shape[0], S.shape[1]])
                    # So, S.todense() returns a numpy.matrix object; a numpy
                    # matrix cannot be converted into a tensor (i.e., added
                    # the third dimension), therefore we need to convert it to
                    # a numpy.array. According to the documentation, the 
                    # attribute .A in a numpy.matrix returns self as an ndarray
                    # object. So that's why the .A is there.
                self.S.append(torch.tensor(S).to(device))
                self.N.append(S.shape[1])
            # And we need to update the GSO in all the places.
            #   Note that we do not need to change the pooling function, because
            #   it is the standard pooling function that doesn't care about the
            #   number of nodes: it still takes one every two of them.
            for l in range(self.L):
                self.GFL[3*l].addGSO(self.S[l]) # Graph convolutional layer
        else:
            # And update in the LSIGF that is still missing (recall that the
            # ordering for the non-coarsening case has already been done)
            for l in range(self.L):
                self.GFL[3*l].addGSO(self.S) # Graph convolutional layer

    def splitForward(self, x):
        
        # Reorder the nodes from the data
        # If we have added dummy nodes (which, has to happen when the size
        # is different and we chose coarsening), then we need to use the
        # provided permCoarsening function (which acts on data to add dummy
        # variables)
        if x.shape[2] != self.N[0] and self.coarsening:
            thisDevice = x.device # Save the device we where operating on
            x = x.cpu().numpy() # Convert to numpy
            x = alegnn.utils.graphTools.permCoarsening(x, self.order)
                # Re order and add dummy values
            x = torch.tensor(x).to(thisDevice)
        else:
        # If not, simply reorder the nodes
            x = x[:, :, self.order]

        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.GFL(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.
    
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        if self.coarsening:
            for l in range(self.L):
                self.S[l] = self.S[l].to(device)
                self.GFL[3*l].addGSO(self.S[l])
        else:
            self.S = self.S.to(device)
            # And all the other variables derived from it.
            for l in range(self.L):
                self.GFL[3*l].addGSO(self.S)
                self.GFL[3*l+2].addGSO(self.S)
                
class LocalActivationGNN(nn.Module):
    """
    LocalActivationGNN: implements the selection GNN architecture with a local
        activation function (instead of pointwise)
        
    Initialization:

        LocalActivationGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                           nonlinearity, kHopActivation, # Nonlinearity
                           nSelectedNodes, poolingFunction, poolingSize, # Pool
                           dimLayersMLP, # MLP in the end
                           GSO, order = None) # Structure

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
            nonlinearity (torch.nn): module from Utils.graphML non-linear
                local activation functions
            kHopActivation (list of int): number of neighborhood hop to include
                in the local activation function at each layer
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            >> Obs.: If coarsening = True, this variable is ignored since the
                number of nodes in each layer is given by the graph coarsening
                algorithm.
            poolingFunction (nn.Module in Utils.graphML or in torch.nn): 
                summarizing function
            >> Obs.: If coarsening = True, then the pooling function is one of
                the regular 1-d pooling functions available in torch.nn (instead
                of one of the summarizing functions in Utils.graphML).
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            >> Obs.: If coarsening = True, then the pooling size is ignored 
                since, due to the binary tree nature of the graph coarsening
                algorithm, it always has to be 2.
                
            /** Readout layers **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        LocalActivationGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                
    Other methods:
        
        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity, kHopActivation,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # kHopActivation is a list with the same number of elements as 
        # nFilterTaps (number of layers)
        assert len(kHopActivation) == len(nFilterTaps)
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.kHop = kHopActivation # k-hop neighborhood for local activation
        self.E = GSO.shape[0] # Number of edge features
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
     
        # Call the corresponding ordering function. Recall that if no
        # order was selected, then this is permIdentity, so that nothing
        # changes.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        self.alpha = poolingSize

        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma(self.kHop[l]))
            # Add GSO for this layer
            gfl[3*l+1].addGSO(self.S)
            #\\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            gfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def changeGSO(self, GSO, nSelectedNodes = [], poolingSize = []):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
        
        # Reorder the new GSO
        self.S, self.order = self.permFunction(GSO)
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # (If it's coarsening, then the pooling size cannot change)
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize
        
        # Now, check if we have a new list of nodes (this only makes sense
        # if there is no coarsening, because if it is coarsening, the list with
        # the number of nodes to be considered is ignored.)
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list 
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.GFL[3*l+2] = self.rho(self.N[l], self.N[l+1],
                                           self.alpha[l])
                self.GFL[3*l+2].addGSO(self.S)
        elif len(nSelectedNodes) == 0 and not self.coarsening:
            # Just update the GSO
            for l in range(self.L):
                self.GFL[3*l+2].addGSO(self.S)
        
        
        # And update in the LSIGF that is still missing (recall that the
        # ordering for the non-coarsening case has already been done)
        for l in range(self.L):
            self.GFL[3*l].addGSO(self.S) # Graph convolutional layer
            self.GFL[3*l+1].addGSO(self.S) # Local Activation function

    def splitForward(self, x):
        
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        y = self.GFL(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.
    
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GFL[3*l].addGSO(self.S)
            self.GFL[3*l+1].addGSO(self.S)
            self.GFL[3*l+2].addGSO(self.S)
                
class LocalGNN(nn.Module):
    """
    LocalGNN: implement the selection GNN architecture where all operations are
        implemented locally, i.e. by means of neighboring exchanges only. More
        specifically, it has graph convolutional layers, but the readout layer,
        instead of being an MLP for the entire graph signal, it is a linear
        combination of the features at each node.
        >> Obs.: This precludes the use of clustering as a pooling operation,
            since clustering is not local (it changes the given graph).

    Initialization:

        LocalGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                 nonlinearity, # Nonlinearity
                 nSelectedNodes, poolingFunction, poolingSize, # Pooling
                 dimReadout, # Local readout layer
                 GSO, order = None # Structure)

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
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            
            /** Readout layers **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Local GNN architecture with the above specified
            characteristics.

    Forward call:

        LocalGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimReadout[-1] x nSelectedNodes[-1]
                
    Other methods:
        
        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimReadout[-1], as well as the output
        of all the GNN layers (i.e. before the readout layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.
        
        y = .singleNodeForward(x, nodes): outputs the value of the last layer
        at a single node. x is the usual input of shape batchSize x dimFeatures
        x numberNodes. nodes is either a single node (int) or a collection of
        nodes (list or numpy.array) of length batchSize, where for each element
        in the batch, we get the output at the single specified node. The
        output y is of shape batchSize x dimReadout[-1].
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.alpha = poolingSize
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimReadout = dimReadout
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma())
            #\\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            gfl[3*l+2].addGSO(self.S)
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
        
    def changeGSO(self, GSO, nSelectedNodes = [], poolingSize = []):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
            
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
        
        # Reorder the new GSO
        self.S, self.order = self.permFunction(GSO)
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize
        
        # Now, check if we have a new list of nodes
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list 
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.GFL[3*l+2] = self.rho(self.N[l], self.N[l+1],
                                           self.alpha[l])
                self.GFL[3*l+2].addGSO(self.S)
        else:
            # Just update the GSO
            for l in range(self.L):
                self.GFL[3*l+2].addGSO(self.S)
        
        # And update in the LSIGF that is still missing
        for l in range(self.L):
            self.GFL[3*l].addGSO(self.S) # Graph convolutional layer

    def splitForward(self, x):

        # Now we compute the forward call
        assert len(x.shape) == 3
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        yGFL = self.GFL(x)
        # Change the order, for the readout
        y = yGFL.permute(0, 2, 1) # B x N[-1] x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x N[-1] x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 2, 1), yGFL
        # B x dimReadout[-1] x N[-1], B x dimFeatures[-1] x N[-1]
    
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output
    
    def singleNodeForward(self, x, nodes):
        
        # x is of shape B x F[0] x N[-1]
        batchSize = x.shape[0]
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
                or type(nodes) is list \
                or type(nodes) is np.ndarray
        
        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x dimRedout[-1] x 1
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
        selectionMatrix = np.zeros([batchSize, self.N[-1], 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype = x.dtype,
                                       device = x.device)
        
        # Now compute the output
        y = self.forward(x)
        # This output is of size B x dimReadout[-1] x N[-1]
        
        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x dimReadout[-1] x 1
        
        # Squeeze the last dimension and return
        return y.squeeze(2)

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GFL[3*l].addGSO(self.S)
            self.GFL[3*l+2].addGSO(self.S)

class SpectralGNN(nn.Module):
    """
    SpectralGNN: implement the selection GNN architecture using spectral filters

    Initialization:

        SpectralGNN(dimNodeSignals, nCoeff, bias, # Graph Filtering
                    nonlinearity, # Nonlinearity
                    nSelectedNodes, poolingFunction, poolingSize, # Pooling
                    dimLayersMLP, # MLP in the end
                    GSO, order = None) # Structure

        Input:
            /** Graph convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nCoeff (list of int): number of coefficients on each layer; if 
                nCoeff[l] is less than the size of the graph, the remaining 
                coefficients are interpolated by means of a cubic spline.
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nCoeff[l] is the number of coefficients for the
                filters implemented at layer l+1, thus len(nCoeff) = L.
                
            
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            
            /** Readout layers **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied

            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics, using filters in the spectral domain.

    Forward call:

        SpectralGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                
    Other methods:
        
        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: This will only work if both the original GSO and the new one
            have the same number of nodes.
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nCoeff, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nCoeff) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nCoeff)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nCoeff)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nCoeff) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.M = nCoeff # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        sgfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            sgfl.append(gml.SpectralGF(self.F[l], self.F[l+1], self.M[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            sgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            sgfl.append(self.sigma())
            #\\ Pooling
            sgfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            sgfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.SGFL = nn.Sequential(*sgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def changeGSO(self, GSO, nSelectedNodes = [], poolingSize = []):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
            
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
        
        # Reorder the new GSO
        self.S, self.order = self.permFunction(GSO)
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize
        
        # Now, check if we have a new list of nodes
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list 
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.SGFL[3*l+2] = self.rho(self.N[l], self.N[l+1],
                                            self.alpha[l])
                self.SGFL[3*l+2].addGSO(self.S)
        else:
            # Just update the GSO
            for l in range(self.L):
                self.SGFL[3*l+2].addGSO(self.S)
        
        # And update in the Spectral GF that is still missing
        for l in range(self.L):
            self.SGFL[3*l].addGSO(self.S) # Graph convolutional layer

    def splitForward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        y = self.SGFL(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.
        
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.SGFL[3*l].addGSO(self.S)
            self.SGFL[3*l+2].addGSO(self.S)

class NodeVariantGNN(nn.Module):
    """
    NodeVariantGNN: implement the selection GNN architecture using node variant
        graph filters

    Initialization:

        NodeVariantGNN(dimNodeSignals, nShiftTaps, nNodeTaps, bias, # Filtering
                       nonlinearity, # Nonlinearity
                       nSelectedNodes, poolingFunction, poolingSize, # Pooling
                       dimLayersMLP, # MLP in the end
                       GSO, order = None) # Structure

        Input:
            /** Graph filtering layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nShiftTaps (list of int): number of shift taps on each layer
                (i.e. information is gathered from up to the (nShiftTaps-1)-hop
                 neighborhood)
            nNodeTaps (list of int): number of node taps on each layer; if 
                nNodesTaps = nNodes, then each node has an independent coeff.
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nShiftTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nShiftTaps) = L.
            >> Obs.: The length of the nShiftTaps and nNodeTaps has to be the
                same, and every element of one list is associated with the
                corresponding one on the other list to create the appropriate
                NVGF filter at each layer.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            
            /** Readout layer **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics, implementing node-variant graph filters.

    Forward call:

        NodeVariantGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                
        Other methods:
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of graph filtering from the effect of the readout
        layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nShiftTaps, nNodeTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than the number of
        # filter taps (because of the input number of features)
        assert len(dimNodeSignals) == len(nShiftTaps) + 1
        # The length of the shift taps list should be equal to the length of the
        # node taps list
        assert len(nShiftTaps) == len(nNodeTaps)
        # nSelectedNodes should be a list of size nShiftTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nShiftTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nShiftTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nShiftTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nShiftTaps # Filter Shift taps
        self.M = nNodeTaps # Filter node taps
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        nvgfl = [] # Node Variant GF Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            nvgfl.append(gml.NodeVariantGF(self.F[l], self.F[l+1],
                                           self.K[l], self.M[l],
                                           self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            nvgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            nvgfl.append(self.sigma())
            #\\ Pooling
            nvgfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            nvgfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.NVGFL = nn.Sequential(*nvgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def splitForward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        y= self.NVGFL(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.
        
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.NVGFL[3*l].addGSO(self.S)
            self.NVGFL[3*l+2].addGSO(self.S)

class EdgeVariantGNN(nn.Module):
    """
    EdgeVariantGNN: implement the selection GNN architecture using edge variant
        graph filters (through masking, not placement)

    Initialization:

        EdgeVariantGNN(dimNodeSignals, nShiftTaps, nFilterNodes, bias,
                       nonlinearity, # Nonlinearity
                       nSelectedNodes, poolingFunction, poolingSize,
                       dimLayersMLP, # MLP in the end
                       GSO, order = None) # Structure

        Input:
            /** Graph filtering layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nShiftTaps (list of int): number of shift taps on each layer
                (i.e. information is gathered from up to the (nShiftTaps-1)-hop
                 neighborhood)
            nFilterNodes (list of int): number of nodes selected for the EV part
                of the hybrid EV filtering (recall that the first ones in the
                given permutation of S are the nodes selected; if any element in
                nFilterNodes is equal to the number of nodes, then we have a
                full edge-variant filter, not an hybrid one)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nShiftTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nShiftTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layer **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics, implementing edge-variant graph filters.

    Forward call:

        EdgeVariantGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                
    Other methods:
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of graph filtering from the effect of the readout
        layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nShiftTaps, nFilterNodes, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than the number of
        # filter taps (because of the input number of features)
        assert len(dimNodeSignals) == len(nShiftTaps) + 1
        # Filter nodes is a list of int with the number of nodes to select for
        # the EV part at each layer; it should have the same length as the
        # number of filter taps
        assert len(nFilterNodes) == len(nShiftTaps)
        # nSelectedNodes should be a list of size nShiftTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nShiftTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nShiftTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nShiftTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nShiftTaps # Filter Shift taps
        self.M = nFilterNodes
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        evgfl = [] # Node Variant GF Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            evgfl.append(gml.EdgeVariantGF(self.F[l], self.F[l+1],
                                            self.K[l], self.M[l], self.N[0],
                                            self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            evgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            evgfl.append(self.sigma())
            #\\ Pooling
            evgfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            evgfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.EVGFL = nn.Sequential(*evgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def splitForward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        y= self.EVGFL(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.
        
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.EVGFL[3*l].addGSO(self.S)
            self.EVGFL[3*l+2].addGSO(self.S)
            
class LocalEdgeNet(nn.Module):
    """
    LocalEdgeNet: implements the selection GNN architecture with edge-variant
        graph filters and local operations only
    Initialization:
        LocalEdgeNet(dimNodeSignals, nShiftTaps, nFilterNodes, bias,
                     nonlinearity, # Nonlinearity
                     nSelectedNodes, poolingFunction, poolingSize,
                     dimReadout, # Local readout layer
                     GSO, order = None) # Structure
        Input:
            /** Graph filtering layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nShiftTaps (list of int): number of shift taps on each layer
                (i.e. information is gathered from up to the (nShiftTaps-1)-hop
                 neighborhood)
            nFilterNodes (list of int): number of nodes selected for the EV part
                of the hybrid EV filtering (recall that the first ones in the
                given permutation of S are the nodes selected; if any element in
                nFilterNodes is equal to the number of nodes, then we have a
                full edge-variant filter, not an hybrid one)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nShiftTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nShiftTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layers **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array
        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics, implementing edge-variant graph filters.
    Forward call:
        LocalEdgeNet(x)
        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes
        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimReadout[-1] x nSelectedNodes[-1]
                
    Other methods:
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of graph filtering from the effect of the readout
        layer.
        
        y = .singleNodeForward(x, nodes): outputs the value of the last layer
        at a single node. x is the usual input of shape batchSize x dimFeatures
        x numberNodes. nodes is either a single node (int) or a collection of
        nodes (list or numpy.array) of length batchSize, where for each element
        in the batch, we get the output at the single specified node. The
        output y is of shape batchSize x dimReadout[-1].
    """
    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nShiftTaps, nFilterNodes, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than the number of
        # filter taps (because of the input number of features)
        assert len(dimNodeSignals) == len(nShiftTaps) + 1
        # Filter nodes is a list of int with the number of nodes to select for
        # the EV part at each layer; it should have the same length as the
        # number of filter taps
        assert len(nFilterNodes) == len(nShiftTaps)
        # nSelectedNodes should be a list of size nShiftTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nShiftTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nShiftTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nShiftTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nShiftTaps # Filter Shift taps
        self.M = nFilterNodes
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimReadout = dimReadout
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        evgfl = [] # Node Variant GF Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            evgfl.append(gml.EdgeVariantGF(self.F[l], self.F[l+1],
                                            self.K[l], self.M[l], self.N[0],
                                            self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            evgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            evgfl.append(self.sigma())
            #\\ Pooling
            evgfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            evgfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.EVGFL = nn.Sequential(*evgfl) # Graph Filtering Layers
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
        
    def splitForward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        yEVGFL = self.EVGFL(x)
        # Change the order, for the readout
        y = yEVGFL.permute(0, 2, 1) # B x N[-1] x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x N[-1] x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 2, 1), yEVGFL
        # B x dimReadout[-1] x N[-1], B x dimFeatures[-1] x N[-1]
        
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output
    
    def singleNodeForward(self, x, nodes):
        
        # x is of shape B x F[0] x N[-1]
        batchSize = x.shape[0]
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
                or type(nodes) is list \
                or type(nodes) is np.ndarray
        
        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x dimRedout[-1] x 1
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
        selectionMatrix = np.zeros([batchSize, self.N[-1], 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype = x.dtype,
                                       device = x.device)
        
        # Now compute the output
        y = self.forward(x)
        # This output is of size B x dimReadout[-1] x N[-1]
        
        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x dimReadout[-1] x 1
        
        # Squeeze the last dimension and return
        return y.squeeze(2)
    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.EVGFL[3*l].addGSO(self.S)
            self.EVGFL[3*l+2].addGSO(self.S)
            
class ARMAfilterGNN(nn.Module):
    """
    ARMAfilterGNN: implements the GNN architecture using ARMA graph filters
        by Jacobi's method.

    Initialization:

        ARMAfilterGNN(dimNodeSignals, nDenominatorTaps,
                      nResidueTaps, bias, # Graph Filtering
                      nonlinearity, # Nonlinearity
                      nSelectedNodes, poolingFunction, poolingSize, # Pooling
                      dimLayersMLP, # MLP in the end
                      GSO, order = None, tMax = 5) # Structure

        Input:
            /** Graph filtering layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nDenominatorTaps (list of int): number of filter taps in the 
                denominator polynomial at each layer
            nResidueTaps (list of int): number of filter taps in the residue
                polynomial at each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nResidueTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nResidueTaps) = L.
                Same holds for nDenominatorTaps.
            
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layer **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array
            
            /** Method specifics **/
            tMax (int): how many iterations in the Jacobi method (default: 5)

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics, implementing ARMA filters.

    Forward call:

        ARMAfilterGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the GNN with
                ARMA filters; shape: batchSize x dimLayersMLP[-1]
                
    Other methods:
            
        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of graph filtering from the effect of the readout
        layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nDenominatorTaps, nResidueTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None, tMax = 5):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nDenominatorTaps) + 1
        assert len(dimNodeSignals) == len(nResidueTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nResidueTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nResidueTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nResidueTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.P = nDenominatorTaps # Denominator taps (order - 1)
        self.K = nResidueTaps # Residue taps (order - 1)
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        self.tMax = tMax
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilterARMA(self.F[l], self.F[l+1],
                                           self.P[l], self.K[l],
                                           self.E, self.bias, self.tMax))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma())
            #\\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            gfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.jARMA = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def changeGSO(self, GSO, nSelectedNodes = [], poolingSize = []):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
            
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
        
        # Reorder the new GSO
        self.S, self.order = self.permFunction(GSO)
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize
        
        # Now, check if we have a new list of nodes
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list 
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.jARMA[3*l+2] = self.rho(self.N[l], self.N[l+1],
                                             self.alpha[l])
                self.jARMA[3*l+2].addGSO(self.S)
        else:
            # Just update the GSO
            for l in range(self.L):
                self.jARMA[3*l+2].addGSO(self.S)
        
        # And update in the LSIGF that is still missing
        for l in range(self.L):
            self.jARMA[3*l].addGSO(self.S) # Graph convolutional layer
        
    def splitForward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        y= self.jARMA(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.
        
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.jARMA[3*l].addGSO(self.S)
            self.jARMA[3*l+2].addGSO(self.S)
            
class LocalARMA(nn.Module):
    """
    LocalARMA: implements the selection GNN architecture using ARMA graph filters
        by Jacobi's method and local operations only
    Initialization:
        LocalARMA(dimNodeSignals,
                  nDenominatorTaps, nResidueTaps, bias, # Graph Filtering
                  nonlinearity, # Nonlinearity
                  nSelectedNodes, poolingFunction, poolingSize, # Pooling
                  dimReadout, # MLP in the end
                  GSO, order = None, tMax = 5) # Structure
        Input:
            /** Graph filtering layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nDenominatorTaps (list of int): number of filter taps in the 
                denominator polynomial at each layer
            nResidueTaps (list of int): number of filter taps in the residue
                polynomial at each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nResidueTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nResidueTaps) = L.
                Same holds for nDenominatorTaps.
            
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layer **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).
            
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array
            
            /** Method specifics **/
            tMax (int): how many iterations in the Jacobi method (default: 5)
        Output:
            nn.Module with a Local GNN architecture with the above specified
            characteristics, implementing ARMA filters.
    Forward call:
        LocalARMA(x)
        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes
        Output:
            y (torch.tensor): output data after being processed by the GNN with
                ARMA filters; shape: 
                    batchSize x dimReadout[-1] x nSelectedNodes[-1]
                
    Other methods:
            
        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of graph filtering from the effect of the readout
        layer.
        
        y = .singleNodeForward(x, nodes): outputs the value of the last layer
        at a single node. x is the usual input of shape batchSize x dimFeatures
        x numberNodes. nodes is either a single node (int) or a collection of
        nodes (list or numpy.array) of length batchSize, where for each element
        in the batch, we get the output at the single specified node. The
        output y is of shape batchSize x dimReadout[-1].
    """
    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nDenominatorTaps, nResidueTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 GSO, order = None, tMax = 5):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nDenominatorTaps) + 1
        assert len(dimNodeSignals) == len(nResidueTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nResidueTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nResidueTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nResidueTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.P = nDenominatorTaps # Denominator taps (order - 1)
        self.K = nResidueTaps # Residue taps (order - 1)
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimReadout = dimReadout
        self.tMax = tMax
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilterARMA(self.F[l], self.F[l+1],
                                           self.P[l], self.K[l],
                                           self.E, self.bias, self.tMax))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma())
            #\\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            gfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.jARMA = nn.Sequential(*gfl) # Graph Filtering Layers
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
        
    def changeGSO(self, GSO, nSelectedNodes = [], poolingSize = []):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
            
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
        
        # Reorder the new GSO
        self.S, self.order = self.permFunction(GSO)
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize
        
        # Now, check if we have a new list of nodes
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list 
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.jARMA[3*l+2] = self.rho(self.N[l], self.N[l+1],
                                             self.alpha[l])
                self.jARMA[3*l+2].addGSO(self.S)
        else:
            # Just update the GSO
            for l in range(self.L):
                self.jARMA[3*l+2].addGSO(self.S)
        
        # And update in the LSIGF that is still missing
        for l in range(self.L):
            self.jARMA[3*l].addGSO(self.S) # Graph convolutional layer
        
    def splitForward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        yARMA = self.jARMA(x)
        # Change the order, for the readout
        y = yARMA.permute(0, 2, 1) # B x N[-1] x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x N[-1] x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 2, 1), yARMA
        # B x dimReadout[-1] x N[-1], B x dimFeatures[-1] x N[-1]
        
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output
    
    def singleNodeForward(self, x, nodes):
        
        # x is of shape B x F[0] x N[-1]
        batchSize = x.shape[0]
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
                or type(nodes) is list \
                or type(nodes) is np.ndarray
        
        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x dimRedout[-1] x 1
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
        selectionMatrix = np.zeros([batchSize, self.N[-1], 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype = x.dtype,
                                       device = x.device)
        
        # Now compute the output
        y = self.forward(x)
        # This output is of size B x dimReadout[-1] x N[-1]
        
        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x dimReadout[-1] x 1
        
        # Squeeze the last dimension and return
        return y.squeeze(2)
    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.jARMA[3*l].addGSO(self.S)
            self.jARMA[3*l+2].addGSO(self.S)

class AggregationGNN(nn.Module):
    """
    AggregationGNN: implement the aggregation GNN architecture

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
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array
            maxN (int): maximum number of neighborhood exchanges (default: None)
            
            /** Multiple nodes options (weight sharing) **/
            nNodes (int): number of nodes on which to compute the aggregation 
                GNN (default: 1)
            dimLayersAggMLP (list of int): Once the information at each of the
                nNodes selected is processed, then they are aggregated together
                through this MLP (default: [] empty list); note that using
                this variable makes the architecture non-local.
            >> Obs.: The nodes selected to carry out the aggregation are those
                corresponding to the first elements in the provided GSO.
            >> Obs.: If dimLayersAggMLP = [], the output is of shape
                    batchSize x numberOfFeatures x nNodes
                where the number of features is given by the last element of the
                dimLayersMLP list (or, if this list is empty, by the last 
                number of the dimFeatures list). However, if nNodes = 1, then
                the output is of shape
                    batchSize x numberOfFeatures
                since we understand that the output is expected to be a summary
                of the entire graph signal

        Output:
            nn.Module with an Aggregation GNN architecture with the above
            specified characteristics.

    Forward call:

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                (Obs.: if nNodes > 1 and dimLayersAggMLP = [], then the output
                is another graph signal of shape
                    batchSize x dimLayersMLP[-1] x nNodes)
    """
    def __init__(self,
                 # Graph filtering
                 dimFeatures, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None, maxN = None,
                 # Multiple nodes options
                 nNodes = 1, dimLayersAggMLP = []):
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimFeatures) == len(nFilterTaps) + 1
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of convolutional layers
        self.F = dimFeatures # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0]
        self.bias = bias # Boolean
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        GSO, self.order = self.permFunction(GSO)
        # We need to keep GSO as the numpy version of the GSO and self.S as the
        # torch version; this is because many of the upcoming operations on the
        # GSO to define the structure are still in numpy.
        self.S = GSO.copy()
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize # This acts as both the kernel_size and the
        # stride, so there is no overlap on the elements over which we take
        # the maximum (this is how it works as default)
        self.dimLayersMLP = dimLayersMLP
        self.dimLayersAggMLP = dimLayersAggMLP
        self.nNodes = nNodes # Number of nodes on which to process the GNN
        # Maybe we don't want to aggregate information all the way to the end,
        # but up to some pre-specificed value maxN (for numerical reasons,
        # mostly)
        if maxN is None:
            self.maxN = GSO.shape[1]
        else:
            self.maxN = maxN if maxN < GSO.shape[1] else GSO.shape[1]
        # Let's also record the number of nodes on each layer (L+1, actually)
        self.N = [self.maxN]
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
        # Now, compute the necessary matrix. Recall that we want to build the
        # vector [[x]_{i}, [Sx]_{i}, [S^2x]_{i}, ..., [S^{N-1}x]_{i}] for the 
        # first i=0,...,nNodes-1 elements. But instead of computing the powers
        # of S^k and then keeping the ith row, we will multiply S with a 
        # [delta_i]_i = 1 and 0s elsewhere and keep each result in the row.
        delta = np.zeros([self.E, GSO.shape[1], self.nNodes]) # E x N x nNodes
        for n in range(self.nNodes):
            delta[:, n, n] = 1. # E x N x nNodes
        # And create the place where to store all of this
        SN = delta.copy().reshape([self.E, 1, GSO.shape[1], self.nNodes])
        for k in range(1, self.maxN):
            delta = GSO @ delta # E x N x nNodes
            SN = np.concatenate((SN,
                                 delta\
                              .reshape([self.E, 1, GSO.shape[1], self.nNodes])),
                                axis = 1) # E x k x N x nNodes
        # Now, we have constructed the matrix E x maxN x N x nNodes, but we want
        # is that signal, when multiplied by this matrix, constructs the vector
        # z for each of the nNodes. This vector z is a map between the N-vector
        # signal and a maxN-vector z, so we want to map N to maxN linearly,
        # multiplying by the left. Therefore, we want a N x maxN matrix. So we
        # reshape the dimensions
        SN = SN.transpose(3, 0, 2, 1) # nNodes x E x N x maxN
        # This matrix SN just needs to multiply the incoming x to obtain the
        # aggregated vector. And that's it.
        self.SN = torch.tensor(SN)
        # The idea to handle different features and different nodes with the 
        # same 1D convolution is realizing that: for each edge feature E we need
        # a different filter, and for each node we need _the same_ convolutional
        # filters. Therefore, the different nNodes will go to increase the
        # batch size, while the edge features will go to increase the feature
        # space. And since different edge features increase the feature space
        # we need to consider them now.
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
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1] * self.E
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done within each node
        self.MLP = nn.Sequential(*fc)
        # Now let's aggregate the information from all nodes
        aggfc = []
        if len(self.dimLayersAggMLP) > 0:
            # If there's a final aggregation layer, then it will have to mix
            # the number of features of each of the nNodes. Note that these
            # number of features will be the output of the last layer of the
            # MLP if there was one, or the last number of features if not
            dimInputAggMLP = dimLayersMLP[-1] if len(dimLayersMLP) > 0 \
                                else self.N[-1] * self.F[-1] * self.E
            # This is the input dimension for each node. So now we need to 
            # multiply this by the number of nodes
            aggfc.append(nn.Linear(dimInputAggMLP * nNodes, dimLayersAggMLP[0],
                                   bias = self.bias))
            # And then, for the rest of the layers
            for l in range(len(dimLayersAggMLP)-1):
                # Add the nonlinearity
                aggfc.append(self.sigma())
                # And the linear layer
                aggfc.append(nn.Linear(dimLayersAggMLP[l], dimLayersAggMLP[l+1],
                                       bias = self.bias))
        # so we finally have the architecture.
        self.AggMLP = nn.Sequential(*aggfc)

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        B = x.shape[0] # batch size
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.SN.shape[2]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # So, up to here, we have:
        #   x of shape B x F x N
        F = x.shape[1]
        N = x.shape[2]
        #   SN of shape nNodes x E x N x maxN
        nNodes = self.SN.shape[0]
        E = self.SN.shape[1]
        maxN = self.SN.shape[3]
        # We will consider a target shape of B x nNodes x E x F x N, so we adapt
        x = x.reshape([B, 1, 1, F, N])
        SN = self.SN.reshape([1, nNodes, E, N, maxN])
        # Let's do the aggregation step
        z = torch.matmul(x, SN) # B x nNodes x E x F x maxN
        # And now, we need to join dimension 0 and 1 (batch and nNodes), and
        # dimensions 2 and 3 (edge features and node features) before feeding
        # this into the convolution as a three-dimensional vector.
        # And since we always join the last dimensions
        z = z.permute(2, 3, 4, 0, 1).reshape([E, F, maxN, B * nNodes])
        z = z.permute(3, 2, 0, 1).reshape([B * nNodes, maxN, E * F])
        z = z.permute(0, 2, 1) # (B * nNodes) x (E * F) x maxN 
        # Let's call the convolutional layers
        y = self.ConvLayers(z)
        # Flatten the output
        y = y.reshape([B * self.nNodes, self.F[-1] * self.N[-1] * self.E])
        # And, feed it into the per node MLP
        y = self.MLP(y) # (B * nNodes) x dimLayersMLP[-1]
        # And now we have to unpack it back for every node
        y = y.permute(1, 0).reshape([y.shape[1], B, nNodes]).permute(1, 0, 2)
        # So that, so far, y is a graph signal (as expected) and as such, has
        # shape B x dimLayersMLP[-1] x nNodes
        # And now, if we have to aggregate one last time, this time we cannot
        # just feed it in the aggregator MLP, because if we're to do so,
        # we need to reshape, but if there's no aggregator, then the output
        # has to be the graph signal, so there's no need for a reshape
        if nNodes == 1 or len(self.dimLayersAggMLP) > 0:
            y = y.reshape([B, y.shape[1] * nNodes])
        y = self.AggMLP(y)
        # And now we're done
        return y

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move to device the GSO and its related variables.
        self.S = self.S.to(device)
        self.SN = self.SN.to(device)
        
class MultiNodeAggregationGNN(nn.Module):
    """
    MultiNodeAggregationGNN: implement the multi-node aggregation GNN
        architecture

    Initialization:

        Input:
            /** External operation: Neighboring exchanges **/
            nSelectedNodes (list of int): number of selected nodes on each
                outer layer
            nShifts (list of int): number of shifts to be done by the selected
                nodes on each outer layer
                
            /** Internal operation: Regular convolution **/
            dimFeatures (list of list of int): the external list corresponds to
                the outer layers, the inner list to how many features to process
                on each inner layer (the aggregation GNN on each node)
            nFilterTaps (list of list of int): the external list corresponds to
                the outer layers, the inner list to how many filter taps to
                consider on each inner layer (the aggregation GNN on each node)
            bias (bool): include bias after graph filter on every layer
            
            /** Internal operation: Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Internal operation: Pooling **/
            poolingFunction (torch.nn): module from torch.nn pooling layers
            poolingSize (list of list of int): the external list corresponds to
                the outer layers, the inner list to the size of the neighborhood
                to compute the summary from at each inner layer (the aggregation
                GNN on each node)
                
            /** Readout layer **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after all the outer layers
                have been computed; note that using this layer makes the
                whole architecture nonlinear.
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Multi-Node Aggregation GNN architecture with the
            above specified characteristics.

    Forward call:

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the 
                multi-node aggregation GNN; shape:
                batchSize x dimLayersMLP[-1]
    
    Example:
        
    We want to create a Multi-Node GNN with two outer layers (i.e. two rounds of
    exchanging information on the graph). In the first round, we select 10 
    nodes, and in the following round, we select 5. Then, we need to determine
    how many shifts (how further away) we are going to move information around.
    In the first round (first outer layer) we shift around 4 times, and in the
    second round, we shift around 8 times (i.e. we get info from up to the 
    4-hop neighborhood in the first round, and 8-hop neighborhood in the
    secound round.)

    nSelectedNodes = [10, 5]
    nShifts = [4, 8]
    
    At this point, we have finished determining the outer structure (the one
    that involves exchanging information with neighbors). Now, we need to
    determine how to process the data within each node (the aggregation GNN
    that happens at each node). Since we have two outer layers, each of these
    parameters will be a list containing two lists. Each of these two lists
    determines the parameters to use to process internally the data. All nodes
    will use the same structure during each round.
    
    Say that we step inside a single node. We start with the signal received
    at the first outer layer (r=0), i.e., we have a signal of length 
    nShifts[0] = 4. We want to process this signal with a two-layer CNN creating
    3 and 6 features, respectively, using 2 filter taps, and with a ReLU
    nonlinearity in between and a max-pooling of size 2. This will just give
    an output with 6 features. This processing occurs at all 
    nSelectedNodes[0] = 10 nodes. After the second round, we get a new signal,
    with 6 features, but of length nShifts[1] = 8 at each of the
    nSelectedNodes[1] = 5 nodes. Now we want to process it through a two-layer
    CNN with that creates 12 and 18 features, with filters of size 2, with
    ReLU nonlinearities (same as before) and a max pooling (same as before) of 
    size 2. The setting is
    
    dimFeatures = [[1, 3, 6], [6, 12, 18]]
    nFilterTaps = [[2, 2], [2, 2]]
    nonlinearity = nn.ReLU
    poolingFunction = nn.MaxPool1d
    poolingSize = [[2, 2], [2, 2]]
    
    Recall that between the last convolutional layer (internal) and the output
    to be shared across nodes, there is an MLP layer adapting the number of
    features to the expected number of features of the next layer.
    
    Once we have all dimFeatures[-1][-1] = 18 features, collected at all
    nSelectedNodes[-1] = 5, we collect this information in a vector and feed it
    through two fully-connected layers of size 20 and 10.
    
    dimLayersMLP = [20, 10]
    """
    def __init__(self,
                 # Outer Structure
                 nSelectedNodes, nShifts,
                 # Inner Structure
                 #  Graph filtering
                 dimFeatures, nFilterTaps, bias,
                 #  Nonlinearity
                 nonlinearity,
                 #  Pooling
                 poolingFunction, poolingSize,
                 #  MLP in the end
                 dimLayersMLP,
                 # Graph Structure
                 GSO, order = None):
        # Initialize parent class
        super().__init__()
        # Check that we have an adequate GSO
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        # And create a third dimension if necessary
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        self.N = GSO.shape[1] # Store the number of nodes
        # Now, the interesting thing is that dimFeatures, nFilterTaps, and
        # poolingSize are all now lists of lists, and all of them need to have
        # the same length.
        self.R = len(nSelectedNodes) # Number of outer layers
        self.P = nSelectedNodes # Number of nodes selected on each outer layer
        # Check that the number of selected nodes does not exceed the number
        # of total nodes.
        # TODO: Should we consider that the number of nodes might not be
        # nonincreasing?
        for r in range(self.R):
            if self.P[r] > self.N:
                # If so, just force it to be the number of nodes.
                self.P[r] = self.N
        assert len(nShifts) == self.R
        self.Q = nShifts # Number of shifts of each node on each outer layer
        assert len(dimFeatures) == len(nFilterTaps) == self.R
        assert len(poolingSize) == self.R
        self.F = dimFeatures # List of lists containing the number of features
            # at each inner layer of each outer layer
        # Note that we have to add how many features we want in the ``last''
        # AggGNN layer before going into the MLP layer. Here, I will just
        # mix in the number of last specified features, but there are a lot of
        # other options, like no MLP whatsoever at the end of each convolutional
        # layer. But, why not?
        # TODO: (This adds quite the number of parameters, it would be nice to
        # do some reasonable tests to check whether this MLPs are necessary or
        # not).
        self.F.append([dimFeatures[-1][-1]])
        self.K = nFilterTaps # List of lists containing the number of filter 
            # taps at each inner layer of each outer layer.
        self.bias = bias # Boolean to include bias or not
        self.sigma = nonlinearity # Pointwise nonlinear function to include on
            # each aggregation GNN
        self.rho = poolingFunction # To use on every aggregation GNN
        self.alpha = poolingSize # Pooling size on each aggregation GNN
        self.dimLayersMLP = dimLayersMLP # MLP for each inner aggregation GNN
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        GSO, self.order = self.permFunction(GSO)
        # We need to keep GSO as the numpy version of the GSO and self.S as the
        # torch version; this is because many of the upcoming operations on the
        # GSO to define the structure are still in numpy.
        self.S = GSO.copy()
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        # Now that there are several things to do next:
        # - The AggregationGNN module always selects the first node, so if we
        #   want to select the first R, then we have to reorder it ourselves
        #   before adding the GSO to each AggregationGNN structure
        # - A regular python list does not register the parameters of the 
        #   corresponding nn.Module leading to bugs and issues on optimization.
        #   For this the class nn.ModuleList() has been created. Unlike 
        #   nn.Sequential(), this class does not have a forward method, because
        #   they are not supposed to act in a cascade way, just to keep track of
        #   dynamically changing numbers of layers.
        # - Another interesting observation is that, preliminary experiments, 
        #   show that nn.ModuleList() is also capable of handling lists of 
        #   lists. And this is precisely what we need: the first element (the
        #   outer one) corresponds to each outer layer, and each one of these
        #   elements contains another list with the Aggregation GNNs
        #   corresponding to the number of selected nodes on each outer layer.
        
        #\\\ Ordering:
        # So, let us start with the ordering. P (the number of selected nodes)
        # determines how many different orders we need (it's just rotating
        # the indices so that each one of those P is first)
        # The order will be a list of lists, the outer list having as many 
        # elements as maximum of P.
        self.innerOrder = [list(range(self.N))] # This is the order for the
        #   first selected nodes which is, clearly, the identity order
        maxP = max(self.P) # Maximum number of nodes to consider
        for p in range(1, maxP):
            allNodes = list(range(self.N)) # Create a list of all the nodes in
            # order.
            allNodes.remove(p) # Get rid of the element that we need to put
            # first
            thisOrder = [p] #  Take the pth element, put it in a list
            thisOrder.extend(allNodes)
            # extend that list with all other nodes, except for the pth one.
            self.innerOrder.append(thisOrder) # Store this in the order list
        
        #\\\ Aggregation GNN stage:
        self.aggGNNmodules = nn.ModuleList() # List to hold the AggGNN modules
        # Create the inner modules
        for r in range(self.R):
            # Add the list of inner modules
            self.aggGNNmodules.append(nn.ModuleList())
            # And start going through the inner modules
            for p in range(self.P[r]):
                thisGSO = GSO[:,self.innerOrder[p],:][:,:,self.innerOrder[p]]
                # # Reorder the GSO so that the selected node comes first and 
                # is thus selected by the AggGNN module.
                # Create the AggGNN module:
                self.aggGNNmodules[r].append(
                        AggregationGNN(self.F[r], self.K[r], self.bias,
                                       self.sigma,
                                       self.rho, self.alpha[r],
                                       # Now, the number of features in the
                                       # output of this AggregationGNN has to
                                       # be equal to the number of input 
                                       # features required at the next AggGNN
                                       # layer.
                                       [self.F[r+1][0]],
                                       thisGSO, maxN = self.Q[r]))
        # And this should be it for the creation of the AggGNN layers of the
        # MultiNodeAggregationGNN architecture. We move onto one last MLP
        fc = []
        if len(self.dimLayersMLP) > 0:
            # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.P[-1] * self.F[-1][0]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def forward(self, x):
        # Now we compute the forward call
        # Check all relative dimensions
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0][0]
        assert x.shape[2] == self.N
        # Reorder
        x = x[:, :, self.order] # B x F x N
        
        # Create an empty vector to store the output of the AggGNN of each node
        y = torch.empty(0).to(x.device)
        # For each outer layer (except the last one, since in the last one we
        # do not have to zero-pad)
        for r in range(self.R-1):
            # For each node
            for p in range(self.P[r]):
                # Re-order the nodes so that the selected nodes goes first
                xReordered = x[:, :, self.innerOrder[p]]
                # Compute the output of each GNN
                thisOutput = self.aggGNNmodules[r][p](xReordered)
                # Add it to the corresponding nodes
                y = torch.cat((y,thisOutput.unsqueeze(2)), dim = 2)
            # After this, y is of size B x F x P[r], but if we need to keep 
            # going for other outer layers, we need to zero-pad so that we can
            # keep shifting around on the original graph
            if y.shape[2] < self.N:
                # We zero-pad
                zeroPad = torch.zeros(batchSize, y.shape[1], self.N-y.shape[2])
                zeroPad = zeroPad.type(y.dtype).to(y.device)
                # Save as x
                x = torch.cat((y, zeroPad), dim = 2)
                # and reset y
                y = torch.empty(0).to(x.device)
                # At this point, note that x (and, before, y) where in order: 
                # the first elements corresponds to the first one in the
                # original ordering and so on. This means that the self.order
                # stored for the MultiNode still holds
            else:
                # We selected all nodes, so we do not need to zero-pad
                x = y
                # Save as x, and reset y
                y = torch.empty(0).to(x.device)
        # Last layer: we do not need to zero pad afterwards, so we just compute
        # the output of the GNN for each node and store that
        for p in range(self.P[-1]):
            xReordered = x[:, :, self.innerOrder[p]]
            thisOutput = self.aggGNNmodules[-1][p](xReordered)
            y = torch.cat((y,thisOutput.unsqueeze(2)), dim = 2)
                
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1][-1] * self.P[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.
    
    def to(self, device):
        # First, we initialize as always.
        super().to(device)
        # And then, in particular, move each architecture (that it will
        # internally move the GSOs and neighbors and stuff)
        for r in range(self.R):
            for p in range(self.P[r]):
                self.aggGNNmodules[r][p].to(device)
            
class GraphAttentionNetwork(nn.Module):
    """
    GraphAttentionNetwork: implement the graph attention network architecture

    Initialization:

        GraphAttentionNetwork(dimNodeSignals, nAttentionHeads, # Graph Filtering
                              nonlinearity, # Nonlinearity
                              nSelectedNodes, poolingFunction, poolingSize,
                              dimLayersMLP, bias, # MLP in the end
                              GSO, order = None) # Structure

        Input:
            /** Attention layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
            nAttentionHeads (list of int): number of attention heads on each
                layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nAttentionHeads[l] is the number of filter taps for
                the filters implemented at layer l+1, thus
                len(nAttentionHeads) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn.functional): function from module
                torch.nn.functional for non-linear activations
                
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layer **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            bias (bool): include bias after each MLP layer
            
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Graph Attention Network architecture with the
            above specified characteristics.

    Forward call:

        GraphAttentionNetwork(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph attentional layer
                 dimNodeSignals, nAttentionHeads,
                 # Nonlinearity (nn.functional)
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP, bias,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nAttentionHeads) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nAttentionHeads)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nAttentionHeads)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nAttentionHeads) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nAttentionHeads # Attention Heads
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlinearity # This has to be a nn.functional instead of
            # just a nn
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        self.bias = bias
        # And now, we're finally ready to create the architecture:
        #\\\ Graph Attentional Layers \\\
        # OBS.: The last layer has to have concatenate False, whereas the rest
        # have concatenate True. So we go all the way except for the last layer
        gat = [] # Graph Attentional Layers
        if self.L > 1:
            # First layer (this goes separate because there are not attention
            # heads increasing the number of features)
            #\\ Graph attention stage:
            gat.append(gml.GraphAttentional(self.F[0], self.F[1], self.K[0],
                                            self.E, self.sigma, True))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[0], self.N[1], self.alpha[0]))
            gat[1].addGSO(self.S)
            # All the next layers (attention heads appear):
            for l in range(1, self.L-1):
                #\\ Graph attention stage:
                gat.append(gml.GraphAttentional(self.F[l] * self.K[l-1],
                                                self.F[l+1], self.K[l],
                                                self.E, self.sigma, True))
                # There is a 2*l below here, because we have two elements per
                # layer: graph filter and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                gat[2*l].addGSO(self.S)
                #\\ Pooling
                gat.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
                # Same as before, this is 2*l+1
                gat[2*l+1].addGSO(self.S)
            # And the last layer (set concatenate to False):
            #\\ Graph attention stage:
            gat.append(gml.GraphAttentional(self.F[self.L-1] * self.K[self.L-2],
                                            self.F[self.L], self.K[self.L-1],
                                            self.E, self.sigma, False))
            gat[2* (self.L - 1)].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[self.L-1], self.N[self.L],
                                self.alpha[self.L-1]))
            gat[2* (self.L - 1) +1].addGSO(self.S)
        else:
            # If there's only one layer, it just go straightforward, adding a
            # False to the concatenation and no increase in the input features
            # due to attention heads
            gat.append(gml.GraphAttentional(self.F[0], self.F[1], self.K[0],
                                            self.E, self.sigma, False))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[0], self.N[1], self.alpha[0]))
            gat[1].addGSO(self.S)
        # And now feed them into the sequential
        self.GAT = nn.Sequential(*gat) # Graph Attentional Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            # NOTE: Because sigma is a functional, instead of the layer, then
            # we need to pick up the layer for the MLP part.
            if str(self.sigma).find('relu') >= 0:
                self.sigmaMLP = nn.ReLU()
            elif str(self.sigma).find('tanh') >= 0:
                self.sigmaMLP = nn.Tanh()
                
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigmaMLP())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph attentional layers
        y = self.GAT(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GAT[2*l].addGSO(self.S)
            self.GAT[2*l+1].addGSO(self.S)

class GraphConvolutionAttentionNetwork(nn.Module):
    """
    GraphConvolutionAttentionNetwork: implement the graph convolution attention
        network (GCAT) architecture

    Initialization:

        GraphConvolutionAttentionNetwork(dimNodeSignals, 
                                         nFilterTaps,
                                         nAttentionHeads,
                                         bias, # Graph Filtering
                                         nonlinearity, # Nonlinearity
                                         nSelectedNodes,
                                         poolingFunction, 
                                         poolingSize,
                                         dimLayersMLP, # MLP in the end
                                         GSO, order = None) # Structure

        Input:
            /** Graph attention convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
            nFilterTaps (list of int): number of filter taps on each layer
            nAttentionHeads (list of int): number of attention heads on each
                layer
            bias (bool): include bias after the graph filter on each layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nAttentionHeads[l] is the number of filter taps for
                the filters implemented at layer l+1, thus
                len(nAttentionHeads) = L. Same for len(nFilterTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn.functional): function from module
                torch.nn.functional for non-linear activations
                
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layer **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Graph Convolutional Attention Network architecture
            with the above specified characteristics.

    Forward call:

        GraphConvolutionAttentionNetwork(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph attentional layer
                 dimNodeSignals, nFilterTaps, nAttentionHeads, bias,
                 # Nonlinearity (nn.functional)
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilterTaps
        # and nAttentionHeads
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        assert len(dimNodeSignals) == len(nAttentionHeads) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nAttentionHeads)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nAttentionHeads)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nAttentionHeads) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Number of filter taps
        self.P = nAttentionHeads # Attention Heads
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlinearity # This has to be a nn.functional instead of
            # just a nn
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        self.bias = bias
        # And now, we're finally ready to create the architecture:
        #\\\ Graph Attentional Layers \\\
        # OBS.: The last layer has to have concatenate False, whereas the rest
        # have concatenate True. So we go all the way except for the last layer
        gat = [] # Graph Attentional Layers
        if self.L > 1:
            # First layer (this goes separate because there are not attention
            # heads increasing the number of features)
            #\\ Graph attention stage:
            gat.append(gml.GraphFilterAttentional(self.F[0],
                                                  self.F[1],
                                                  self.K[0],
                                                  self.P[0],
                                                  self.E,
                                                  self.bias,
                                                  self.sigma,
                                                  True))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[0], self.N[1], self.alpha[0]))
            gat[1].addGSO(self.S)
            # All the next layers (attention heads appear):
            for l in range(1, self.L-1):
                #\\ Graph attention stage:
                gat.append(gml.GraphFilterAttentional(self.F[l] * self.P[l-1],
                                                      self.F[l+1],
                                                      self.K[l],
                                                      self.P[l],
                                                      self.E,
                                                      self.bias,
                                                      self.sigma,
                                                      True))
                # There is a 2*l below here, because we have two elements per
                # layer: graph filter and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                gat[2*l].addGSO(self.S)
                #\\ Pooling
                gat.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
                # Same as before, this is 2*l+1
                gat[2*l+1].addGSO(self.S)
            # And the last layer (set concatenate to False):
            #\\ Graph attention stage:
            gat.append(gml.GraphFilterAttentional(self.F[self.L-1] \
                                                             * self.P[self.L-2],
                                                  self.F[self.L],
                                                  self.K[self.L-1],
                                                  self.P[self.L-1],
                                                  self.E,
                                                  self.bias,
                                                  self.sigma,
                                                  False))
            gat[2* (self.L - 1)].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[self.L-1], self.N[self.L],
                                self.alpha[self.L-1]))
            gat[2* (self.L - 1) +1].addGSO(self.S)
        else:
            # If there's only one layer, it just go straightforward, adding a
            # False to the concatenation and no increase in the input features
            # due to attention heads
            gat.append(gml.GraphFilterAttentional(self.F[0],
                                                  self.F[1],
                                                  self.K[0],
                                                  self.P[0],
                                                  self.E,
                                                  self.bias,
                                                  self.sigma,
                                                  False))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[0], self.N[1], self.alpha[0]))
            gat[1].addGSO(self.S)
        # And now feed them into the sequential
        self.GCAT = nn.Sequential(*gat) # Graph Attentional Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            # NOTE: Because sigma is a functional, instead of the layer, then
            # we need to pick up the layer for the MLP part.
            if str(self.sigma).find('relu') >= 0:
                self.sigmaMLP = nn.ReLU()
            elif str(self.sigma).find('tanh') >= 0:
                self.sigmaMLP = nn.Tanh()
                
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigmaMLP())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph attentional layers
        y = self.GCAT(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GCAT[2*l].addGSO(self.S)
            self.GCAT[2*l+1].addGSO(self.S)
            
class EdgeVariantAttention(nn.Module):
    """
    EdgeVariantAttention: implement the edge variant graph filter, with 
        coefficients learned following a parameterization given by the 
        attention mechanism

    Initialization:

        EdgeVariantAttention(dimNodeSignals, nFilterTaps,
                             nAttentionHeads, bias, # Graph Filtering
                             nonlinearity, # Nonlinearity
                             nSelectedNodes, poolingFunction, poolingSize,
                             dimLayersMLP, # MLP in the end
                             GSO, order = None) # Structure

        Input:
            /** Graph attention filtering layer **/
            dimNodeSignals (list of int): dimension of the signals at each layer
            nFilterTaps (list of int): number of filter taps on each layer
            nAttentionHeads (list of int): number of attention heads on each
                layer
            bias (bool): include bias after the graph filter on each layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nAttentionHeads[l] is the number of filter taps for
                the filters implemented at layer l+1, thus
                len(nAttentionHeads) = L. Same for len(nFilterTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn.functional): function from module
                torch.nn.functional for non-linear activations
                
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layer **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with an edge variant graph filter whose coefficients are
            parameterized by an attention mechanism

    Forward call:

        EdgeVariantAttention(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph attentional layer
                 dimNodeSignals, nFilterTaps, nAttentionHeads, bias,
                 # Nonlinearity (nn.functional)
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        assert len(dimNodeSignals) == len(nAttentionHeads) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nAttentionHeads)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nAttentionHeads)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nAttentionHeads) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.P = nAttentionHeads # Attention Heads
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlinearity # This has to be a nn.functional instead of
            # just a nn
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        self.bias = bias
        # And now, we're finally ready to create the architecture:
        #\\\ Graph Attentional Layers \\\
        # OBS.: The last layer has to have concatenate False, whereas the rest
        # have concatenate True. So we go all the way except for the last layer
        gat = [] # Graph Attentional Layers
        if self.L > 1:
            # First layer (this goes separate because there are not attention
            # heads increasing the number of features)
            #\\ Graph attention stage:
            gat.append(gml.EdgeVariantAttentional(self.F[0],
                                                  self.F[1],
                                                  self.K[0],
                                                  self.P[0],
                                                  self.E,
                                                  self.bias,
                                                  self.sigma,
                                                  True))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[0], self.N[1], self.alpha[0]))
            gat[1].addGSO(self.S)
            # All the next layers (attention heads appear):
            for l in range(1, self.L-1):
                #\\ Graph attention stage:
                gat.append(gml.EdgeVariantAttentional(self.F[l]*self.P[l-1],
                                                      self.F[l+1],
                                                      self.K[l],
                                                      self.P[l],
                                                      self.E,
                                                      self.bias,
                                                      self.sigma,
                                                      True))
                # There is a 2*l below here, because we have two elements per
                # layer: graph filter and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                gat[2*l].addGSO(self.S)
                #\\ Pooling
                gat.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
                # Same as before, this is 2*l+1
                gat[2*l+1].addGSO(self.S)
            # And the last layer (set concatenate to False):
            #\\ Graph attention stage:
            gat.append(gml.EdgeVariantAttentional(self.F[self.L-1] \
                                                             * self.K[self.L-2],
                                                  self.F[self.L],
                                                  self.K[self.L-1],
                                                  self.P[self.L-1],
                                                  self.E,
                                                  self.bias,
                                                  self.sigma,
                                                  False))
            gat[2* (self.L - 1)].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[self.L-1], self.N[self.L],
                                self.alpha[self.L-1]))
            gat[2* (self.L - 1) +1].addGSO(self.S)
        else:
            # If there's only one layer, it just go straightforward, adding a
            # False to the concatenation and no increase in the input features
            # due to attention heads
            gat.append(gml.EdgeVariantAttentional(self.F[0],
                                                  self.F[1],
                                                  self.K[0],
                                                  self.P[0],
                                                  self.E,
                                                  self.bias,
                                                  self.sigma,
                                                  False))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[0], self.N[1], self.alpha[0]))
            gat[1].addGSO(self.S)
        # And now feed them into the sequential
        self.EVGAT = nn.Sequential(*gat) # Graph Attentional Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            # NOTE: Because sigma is a functional, instead of the layer, then
            # we need to pick up the layer for the MLP part.
            if str(self.sigma).find('relu') >= 0:
                self.sigmaMLP = nn.ReLU()
            elif str(self.sigma).find('tanh') >= 0:
                self.sigmaMLP = nn.Tanh()
                
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigmaMLP())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph attentional layers
        y = self.EVGAT(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.EVGAT[2*l].addGSO(self.S)
            self.EVGAT[2*l+1].addGSO(self.S)

class GraphRecurrentNN(nn.Module):
# Luana R. Ruiz, rubruiz@seas.upenn.edu, 2021/03/04
    """
    GraphRecurrentNN: implements the GRNN architecture. It is a single-layer 
    GRNN and the hidden state is initialized at random drawing from a standard 
    gaussian.
    
    Initialization:
        
        GraphRecurrentNN(dimInputSignals, dimOutputSignals,
                            dimHiddenSignals, nFilterTaps, bias, # Filtering
                            nonlinearityHidden, nonlinearityOutput,
                            nonlinearityReadout, # Nonlinearities
                            dimReadout, # Local readout layer
                            dimEdgeFeatures, # Structure
                            GSO)
        
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
            GSO (np.array): graph shift operator of choice.
            
        Output:
            nn.Module with a GRNN architecture with the above specified
            characteristics
            
    Forward call:
        
        GraphRecurrentNN(x)
        
        Input:
            x (torch.tensor): input data of shape
                batchSize x timeSamples x dimInputSignals x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the GRNN; 
                batchSize x timeSamples x dimReadout[-1] x numberNodes
        
    Other methods:
            
        y, yGNN = .splitForward(x): gives the output of the entire GRNN y,
        which has shape batchSize x timeSamples x dimReadout[-1] x numberNodes,
        as well as the output of the GRNN (i.e. before the readout layers), 
        yGNN of shape batchSize x timeSamples x dimInputSignals x numberNodes. 
        This can be used to isolate the effect of the graph convolutions from 
        the effect of the readout layer.
        
        y = .singleNodeForward(x, nodes): outputs the value of the last
        layer at a single node. x is the usual input of shape batchSize 
        x timeSamples x dimInputSignals x numberNodes. nodes is either a single
        node (int) or a collection of nodes (list or numpy.array) of length
        batchSize, where for each element in the batch, we get the output at
        the single specified node. The output y is of shape batchSize 
        x timeSamples x dimReadout[-1].
        
        .changeGSO(S): takes as input a new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the GraphRecurrentNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one.
        >> Obs.: The number of nodes in the GSOs need not be the same.
            
        
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
                 GSO):
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
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearityHidden
        self.rho = nonlinearityOutput
        self.nonlinearityReadout = nonlinearityReadout
        self.dimReadout = dimReadout
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        self.E = GSO.shape[0] # Number of edge features
        self.N = GSO.shape[1]
        self.S = GSO
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        
        #\\\ Hidden State RNN \\\
        # Create the layer that generates the hidden state, and generate z0
        self.hiddenState = gml.HiddenState(self.F, self.H, self.K[0],
                                       nonlinearity = self.sigma, E = self.E,
                                       bias = self.bias)
        #\\\ Output Graph Filters \\\
        self.outputState = gml.GraphFilter(self.H, self.G, self.K[1],
                                              E = self.E, bias = self.bias)
        # Add the GSO for each graph filter
        self.hiddenState.addGSO(self.S)
        self.outputState.addGSO(self.S)
        
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
        
    def splitForward(self, x):

        # Check the dimensions of the input
        #   S: E x N x N
        #   x: B x T x F[0] x N
        assert len(self.S.shape) == 3
        assert self.S.shape[0] == self.E     
        N = self.S.shape[1]
        assert self.S.shape[2] == N
        
        assert len(x.shape) == 4
        B = x.shape[0]
        T = x.shape[1]
        assert x.shape[2] == self.F
        assert x.shape[3] == N
        
        # This can be generated here or generated outside of here, not clear yet
        # what's the most coherent option
        z0 = torch.randn((B, self.H, N), device = x.device)
        
        # Compute the trajectory of hidden states
        z, _ = self.hiddenState(x, z0)
        z = z.reshape((B*T,self.H,N))
        # Compute the output trajectory from the hidden states
        yOut = self.outputState(z)
        yOut = self.rho(yOut) # Don't forget the nonlinearity!
        yOut = yOut.reshape((B,T,self.G,N))
        #   B x T x G x N
        # Change the order, for the readout
        y = yOut.permute(0, 1, 3, 2) # B x T x N x G
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x T x N x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 1, 3, 2), yOut
        # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N
    
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output
        
    def singleNodeForward(self, x, nodes):
        
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
        y = self.forward(x)
        # This output is of size B x T x dimReadout[-1] x N
        
        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x T x dimReadout[-1] x 1
        
        # Squeeze the last dimension and return
        return y.squeeze(3)
    
    def changeGSO(self, GSO):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
            
        self.S = GSO
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Add the GSO for each graph filter
        self.hiddenState.addGSO(self.S)
        self.outputState.addGSO(self.S)
            
class GatedGraphRecurrentNN(nn.Module):
# Luana R. Ruiz, rubruiz@seas.upenn.edu, 2021/03/04
    """
   GatedGraphRecurrentNN: implements the (time, node, edge)-gated GRNN 
   architecture. It is a single-layer gated GRNN and the hidden state is 
   initialized at random drawing from a standard gaussian.
    
    Initialization:
        
        GatedGraphRecurrentNN(dimInputSignals, dimOutputSignals,
                            dimHiddenSignals, nFilterTaps, bias, # Filtering
                            nonlinearityHidden, nonlinearityOutput,
                            nonlinearityReadout, # Nonlinearities
                            dimReadout, # Local readout layer
                            dimEdgeFeatures, GSO, # Structure
                            gateType) # Gating
        
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
            GSO (np.array): graph shift operator of choice.
            
            /** Gating **/
            gateType (string): 'time', 'node' or 'edge' gating
            
            
        Output:
            nn.Module with a gated GRNN architecture with the above specified
            characteristics
            
    Forward call:
        
        GatedGraphRecurrentNN(x)
        
        Input:
            x (torch.tensor): input data of shape
                batchSize x timeSamples x dimInputSignals x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the GRNN; 
                batchSize x timeSamples x dimReadout[-1] x numberNodes
        
    Other methods:
            
        y, yGNN = .splitForward(x): gives the output of the entire GRNN y,
        which has shape batchSize x timeSamples x dimReadout[-1] x numberNodes,
        as well as the output of the GRNN (i.e. before the readout layers), 
        yGNN of shape batchSize x timeSamples x dimInputSignals x numberNodes. 
        This can be used to isolate the effect of the graph convolutions from 
        the effect of the readout layer.
        
        y = .singleNodeForward(x, nodes): outputs the value of the last
        layer at a single node. x is the usual input of shape batchSize 
        x timeSamples x dimInputSignals x numberNodes. nodes is either a single
        node (int) or a collection of nodes (list or numpy.array) of length
        batchSize, where for each element in the batch, we get the output at
        the single specified node. The output y is of shape batchSize 
        x timeSamples x dimReadout[-1].
        
        .changeGSO(S): takes as input a new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the GatedGraphRecurrentNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one.
        >> Obs.: The number of nodes in the GSOs need not be the same.
            
        
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
                 GSO,
                 # Gating
                 gateType):
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
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearityHidden
        self.rho = nonlinearityOutput
        self.nonlinearityReadout = nonlinearityReadout
        self.dimReadout = dimReadout
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        self.E = GSO.shape[0] # Number of edge features
        self.N = GSO.shape[1]
        self.S = GSO
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        
        #\\\ Hidden State RNN \\\
        # Create the layer that generates the hidden state, and generate z0
        # The type of hidden layer depends on the type of gating
        assert gateType == 'time' or gateType == 'node' or gateType == 'edge'
        if gateType == 'time':
            self.hiddenState = gml.TimeGatedHiddenState(self.F, self.H, self.K[0],
                                       nonlinearity = self.sigma, E = self.E,
                                       bias = self.bias)
        elif gateType == 'node':
            self.hiddenState = gml.NodeGatedHiddenState(self.F, self.H, self.K[0],
                                       nonlinearity = self.sigma, E = self.E,
                                       bias = self.bias)
        elif gateType == 'edge':
            self.hiddenState = gml.EdgeGatedHiddenState(self.F, self.H, self.K[0],
                                       nonlinearity = self.sigma, E = self.E,
                                       bias = self.bias)
        #\\\ Output Graph Filters \\\
        self.outputState = gml.GraphFilter(self.H, self.G, self.K[1],
                                              E = self.E, bias = self.bias)
        # Add the GSO for each graph filter
        self.hiddenState.addGSO(self.S)
        self.outputState.addGSO(self.S)
        
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
        
    def splitForward(self, x):

        # Check the dimensions of the input
        #   S: E x N x N
        #   x: B x T x F[0] x N
        assert len(self.S.shape) == 3
        assert self.S.shape[0] == self.E     
        N = self.S.shape[1]
        assert self.S.shape[2] == N
        
        assert len(x.shape) == 4
        B = x.shape[0]
        T = x.shape[1]
        assert x.shape[2] == self.F
        assert x.shape[3] == N
        
        # This can be generated here or generated outside of here, not clear yet
        # what's the most coherent option
        z0 = torch.randn((B, self.H, N), device = x.device)
        
        # Compute the trajectory of hidden states
        z, _ = self.hiddenState(x, z0)
        z = z.reshape((B*T,self.H,N))
        # Compute the output trajectory from the hidden states
        yOut = self.outputState(z)
        yOut = self.rho(yOut) # Don't forget the nonlinearity!
        yOut = yOut.reshape((B,T,self.G,N))
        #   B x T x G x N
        # Change the order, for the readout
        y = yOut.permute(0, 1, 3, 2) # B x T x N x G
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x T x N x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 1, 3, 2), yOut
        # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N
    
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output
        
    def singleNodeForward(self, x, nodes):
        
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
        y = self.forward(x)
        # This output is of size B x T x dimReadout[-1] x N
        
        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x T x dimReadout[-1] x 1
        
        # Squeeze the last dimension and return
        return y.squeeze(3)
    
    def changeGSO(self, GSO):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
            
        self.S = GSO
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Add the GSO for each graph filter
        self.hiddenState.addGSO(self.S)
        self.outputState.addGSO(self.S)
        