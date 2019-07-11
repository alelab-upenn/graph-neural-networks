# 2018/11/01~2018/07/12
# Fernando Gama, fgama@seas.upenn.edu.
"""
graphML.py Module for basic GSP and graph machine learning functions.

Functionals

LSIGF: Applies a linear shift-invariant graph filter
spectralGF: Applies a linear shift-invariant graph filter in spectral form
NVGF: Applies a node-variant graph filter
EVGF: Applies an edge-variant graph filter
learnAttentionGSO: Computes the GSO following the attention mechanism
graphAttention: Applies a graph attention layer

Filtering Layers (nn.Module)

GraphFilter: Creates a graph convolutional layer using LSI graph filters
SpectralGF: Creates a graph convolutional layer using LSI graph filters in
    spectral form
NodeVariantGF: Creates a graph filtering layer using node-variant graph filters
EdgeVariantGF: Creates a graph filtering layer using edge-variant graph filters
GraphAttentional: Creates a layer using graph attention mechanisms

Activation Functions - Nonlinearities (nn.Module)

MaxLocalActivation: Creates a localized max activation function layer
MedianLocalActivation: Creates a localized median activation function layer
NoActivation: Creates a layer for no activation function

Summarizing Functions - Pooling (nn.Module)

NoPool: No summarizing function.
MaxPoolLocal: Max-summarizing function
"""

import math
import numpy as np
import torch
import torch.nn as nn

import Utils.graphTools as graphTools

zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number

# WARNING: Only scalar bias.

def LSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[0] == E
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        x = torch.matmul(x, S) # B x E x G x N
        xS = x.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E*K*G]),
                     h.reshape([F, E*K*G]).permute(1, 0)).permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

def spectralGF(h, V, VH, x, b=None):
    """
    spectralGF(filter_coeff, eigenbasis, eigenbasis_hermitian, input, bias=None)
        Computes the output of a linear shift-invariant graph filter in spectral
        form applying filter_coefficients on the graph fourier transform of the
        input .

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, N the number of nodes, S_{e} in R^{N x N}
    the GSO for edge feature e with S_{e} = V_{e} Lambda_{e} V_{e}^{H} as
    eigendecomposition, x in R^{G x N} the input data where x_{g} in R^{N} is
    the graph signal representing feature g, and b in R^{F x N} the bias vector,
    with b_{f} in R^{N} representing the bias for feature f.

    Then, the LSI-GF in spectral form is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{g=1}^{G}
                        V_{e} diag(h_{f,g,e}) V_{e}^{H} x_{g}
                + b_{f}
    for f = 1, ..., F, with h_{f,g,e} in R^{N} the filter coefficients for
    output feature f, input feature g and edge feature e.

    Inputs:
        filter_coeff (torch.tensor): array of filter coefficients; shape:
            output_features x edge_features x input_features x number_nodes
        eigenbasis (torch.tensor): eigenbasis of the graph shift operator;shape:
            edge_features x number_nodes x number_nodes
        eigenbasis_hermitian (torch.tensor): hermitian of the eigenbasis; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes

    Obs.: While we consider most GSOs to be normal (so that the eigenbasis is
    an orthonormal basis), this function would also work if V^{-1} is used as
    input instead of V^{H}
    """
    # The decision to input both V and V_H is to avoid any time spent in
    # permuting/inverting the matrix. Because this depends on the graph and not
    # the data, it can be done faster if we just input it.

    # h is output_features x edge_weights x input_features x number_nodes
    # V is edge_weighs x number_nodes x number_nodes
    # VH is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    G = h.shape[2]
    N = h.shape[3]
    assert V.shape[0] == VH.shape[0] == E
    assert V.shape[1] == VH.shape[1] == V.shape[2] == VH.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation I've been using:
    # h in F x E x G x N
    # V in E x N x N
    # VH in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # We will do proper matrix multiplication in this case (algebraic
    # multiplication using column vectors instead of CS notation using row
    # vectors).
    # We will multiply separate VH with x, and V with diag(h).
    # First, to multiply VH with x, we need to add one dimension for each one
    # of them (dimension E for x and dimension B for VH)
    x = x.reshape([B, 1, G, N]).permute(0, 1, 3, 2) # B x 1 x N x G
    VH = VH.reshape([1, E, N, N]) # 1 x E x N x N
    # Now we multiply. Note that we also permute to make it B x E x G x N
    # instead of B x E x N x G because we want to multiply for a specific e and
    # g, there we do not want to sum (yet) over G.
    VHx = torch.matmul(VH, x).permute(0, 1, 3, 2) # B x E x G x N

    # Now we want to multiply V * diag(h), both are matrices. So first, we
    # add the necessary dimensions (B and G for V and an extra N for h to make
    # it a matrix from a vector)
    V = V.reshape([1, E, 1, N, N]) # 1 x E x 1 x N x N
    # We note that multiplying by a diagonal matrix to the right is equivalent
    # to an elementwise multiplication in which each column is multiplied by
    # a different number, so we will do this to make it faster (elementwise
    # multiplication is faster than matrix multiplication). We need to repeat
    # the vector we have columnwise.
    diagh = h.reshape([F, E, G, 1, N]).repeat(1, 1, 1, N, 1) # F x E x G x N x N
    # And now we do elementwise multiplication
    Vdiagh = V * diagh # F x E x G x N x N
    # Finally, we make the multiplication of these two matrices. First, we add
    # the corresponding dimensions
    Vdiagh = Vdiagh.reshape([1, F, E, G, N, N]) # 1 x F x E x G x N x N
    VHx = VHx.reshape([B, 1, E, G, N, 1]) # B x 1 x E x G x N x 1
    # And do matrix multiplication to get all the corresponding B,F,E,G vectors
    VdiaghVHx = torch.matmul(Vdiagh, VHx) # B x F x E x G x N x 1
    # Get rid of the last dimension which we do not need anymore
    y = VdiaghVHx.squeeze(5) # B x F x E x G x N
    # Sum over G
    y = torch.sum(y, dim = 3) # B x F x E x N
    # Sum over E
    y = torch.sum(y, dim = 2) # B x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

def NVGF(h, S, x, b=None):
    """
    NVGF(filter_taps, GSO, input, bias=None) Computes the output of a
    node-variant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of shifts, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f. Denote as h_{k}^{efg} in R^{N} the vector with the N
    filter taps corresponding to the efg filter for shift k.

    Then, the NV-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        diag(h_{k}^{efg}) S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
                x number_nodes
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # h is output_features x edge_weights x filter_taps x input_features
    #                                                             x number_nodes
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    N = h.shape[4]
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation I've been using:
    # h in F x E x K x G x N
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    xr = x.reshape([B, 1, G, N])
    Sr = S.reshape([1, E, N, N])
    z = xr.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        xr = torch.matmul(xr, Sr) # B x E x G x N
        xS = xr.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # This multiplication with filter taps is ``element wise'' on N since for
    # each node we have a different element
    # First, add the extra dimension (F for z, and B for h)
    z = z.reshape([B, 1, E, K, G, N])
    h = h.reshape([1, F, E, K, G, N])
    # Now let's do elementwise multiplication
    zh = z * h
    # And sum over the dimensions E, K, G to get B x F x N
    y = torch.sum(zh, dim = 4) # Sum over G
    y = torch.sum(y, dim = 3) # Sum over K
    y = torch.sum(y, dim = 2) # Sum over E
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

def EVGF(S, x, b=None):
    """
    EVGF(filter_matrices, input, bias=None) Computes the output of an
    edge-variant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of shifts, N the number of
    nodes, Phi_{efg} in R^{N x N} the filter matrix for edge feature e, output
    feature f and input feature g (recall that Phi_{efg}^{k} has the same
    sparsity pattern as the graph, except for Phi_{efg}^{0} which is expected to
    be a diagonal matrix), x in R^{G x N} the input data where x_{g} in R^{N} is
    the graph signal representing feature g, and b in R^{F x N} the bias vector,
    with b_{f} in R^{N} representing the bias for feature f.

    Then, the EV-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        Phi_{efg}^{k:0} x_{g}
                + b_{f}
    for f = 1, ..., F, with Phi_{efg}^{k:0} = Phi_{efg}^{k} Phi_{efg}^{k-1} ...
    Phi_{efg}^{0}.

    Inputs:
        filter_matrices (torch.tensor): array of filter matrices; shape:
            output_features x edge_features x filter_taps x input_features
                x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # We just need to multiply by the filter_matrix recursively, and then
    # add for all E, G, and K features.

    # S is output_features x edge_features x filter_taps x input_features
    #   x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = S.shape[0]
    E = S.shape[1]
    K = S.shape[2]
    G = S.shape[3]
    N = S.shape[4]
    assert S.shape[5] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation I've been using:
    # S in F x E x K x G x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # We will be doing matrix multiplications in the algebraic way, trying to
    # multiply the N x N matrix corresponding to the appropriate e, f, k and g
    # dimensions, with the respective x vector (N x 1 column vector)
    # For this, we first add the corresponding dimensions (for x we add
    # dimensions F, E and the last dimension for column vector)
    x = x.reshape([B, 1, 1, G, N, 1])
    # When we do index_select along dimension K we get rid of this dimension
    Sk = torch.index_select(S, 2, torch.tensor(0).to(S.device)).squeeze(2)
    # Sk in F x E x G x N x N
    # And we add one further dimension for the batch size B
    Sk = Sk.unsqueeze(0) # 1 x F x E x G x N x N
    # Matrix multiplication
    x = torch.matmul(Sk, x) # B x F x E x G x N x 1
    # And we collect this for every k in a vector z, along the K dimension
    z = x.reshape([B, F, E, 1, G, N, 1]).squeeze(6) # B x F x E x 1 x G x N
    # Now we do all the matrix multiplication
    for k in range(1,K):
        # Extract the following k
        Sk = torch.index_select(S, 2, torch.tensor(k).to(S.device)).squeeze(2)
        # Sk in F x E x G x N x N
        # Give space for the batch dimension B
        Sk = Sk.unsqueeze(0) # 1 x F x E x G x N x N
        # Multiply with the previously cumulative Sk * x
        x = torch.matmul(Sk, x) # B x F x E x G x N x 1
        # Get rid of the last dimension (of a column vector)
        Sx = x.reshape([B, F, E, 1, G, N, 1]).squeeze(6) # B x F x E x 1 x G x N
        # Add to the z
        z = torch.cat((z, Sx), dim = 2) # B x F x E x k x G x N
    # Sum over G
    z = torch.sum(z, dim = 4)
    # Sum over K
    z = torch.sum(z, dim = 3)
    # Sum over E
    y = torch.sum(z, dim = 2)
    if b is not None:
        y = y + b
    return y  

def learnAttentionGSO(x, a, W, S, negative_slope=0.2):
    """
    learnAttentionGSO(x, a, W, S) Computes the GSO following the attention
        mechanism

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.
    
    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    B = x.shape[0] # batch_size
    G = x.shape[1] # input_features
    N = x.shape[2] # number_nodes
    P = a.shape[0] # number_heads
    E = a.shape[1] # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2] # output_features
    assert a.shape[2] == int(2*F)
    G = W.shape[3] # input_features
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    S = S + torch.eye(N).reshape([1,N,N]).repeat(E,1,1).to(S.device)
    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, P, E, F, G])
    Wx = torch.matmul(W, x) # B x P x E x F x N
    # Now, do a_1^T Wx, and a_2^T Wx to get a tensor of shape B x P x E x 1 x N
    # because we're applying the inner product on the F dimension.
    a1 = torch.index_select(a, 2, torch.arange(F).to(x.device)) # K x E x F
    a2 = torch.index_select(a, 2, torch.arange(F, 2*F).to(x.device)) # K x E x F
    a1Wx = torch.matmul(a1.reshape([1, P, E, 1, F]), Wx) # B x P x E x 1 x N
    a2Wx = torch.matmul(a2.reshape([1, P, E, 1, F]), Wx) # B x P x E x 1 x N
    # And then, use this to sum them accordingly and create a B x P x E x N x N
    # matrix.
    aWx = a1Wx + a2Wx.permute(0, 1, 2, 4, 3) # B x P x E x N x N
    #   Obs.: In this case, we have one column vector and one row vector; then,
    # what the sum does, is to repeat the column and the row, respectively,
    # until both matrices are of the same size, and then adds up, which is
    # precisely what we want to do
    # Apply the LeakyRelu
    eij = nn.functional.leaky_relu(aWx, negative_slope = negative_slope)
    #   B x P x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim = 0)
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype)
    #   Make it -infinity where there are zeros
    infinityMask = (1-maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    aij = nn.functional.softmax(eij*maskEdges - infinityMask, dim = 4)
    #   B x P x E x N x N
    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    return aij * maskEdges # B x P x E x N x N

def graphAttention(x, a, W, S, negative_slope=0.2):
    """
    graphAttention(x, a, W, S) Computes attention following GAT layer taking
        into account multiple edge features.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Let y_{i}^{p} in R^{F} be the output of the graph attention at node i for
    attention head p. It is computed as
        y_{i}^{p} = \sum_{e=1}^{E}
                        \sum_{j in N_{i}}
                            s_{ij}^{e} alpha_{ij}^{ep} W^{ep} x_{j}
    with
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        y: output; shape:
            batch_size x number_heads x output_features x number_nodes
    """
    B = x.shape[0] # batch_size
    G = x.shape[1] # input_features
    N = x.shape[2] # number_nodes
    P = a.shape[0] # number_heads
    E = a.shape[1] # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2] # output_features
    assert a.shape[2] == int(2*F)
    G = W.shape[3] # input_features
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N

    # First, we need to learn the attention GSO
    aij = learnAttentionGSO(x, a, W, S, negative_slope = negative_slope)
    # B x P x E x N x N

    # Then, we need to compute the high-level features
    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, P, E, F, G])
    Wx = torch.matmul(W, x) # B x P x E x F x N
    
    # Finally, we just need to apply this matrix to the Wx which we have already
    # computed, and done.
    y = torch.matmul(Wx, S.reshape([1, 1, E, N, N]) * aij) # B x P x E x F x N
    # And sum over all edges
    return torch.sum(y, dim = 2) # B x P x F x N

class MaxLocalActivation(nn.Module):
    # Luana R. Ruiz, rubruiz@seas.upenn.edu, 2019/03/15
    """
    MaxLocalActivation creates a localized activation function layer on graphs

    Initialization:

        MaxLocalActivation(K)

        Inputs:
            K (int): number of hops (>0)

        Output:
            torch.nn.Module for a localized max activation function layer

    Add graph shift operator:

        MaxLocalActivation.addGSO(GSO) Before applying the filter, we need to
        define the GSO that we are going to use. This allows to change the GSO
        while using the same filtering coefficients (as long as the number of
        edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = MaxLocalActivation(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes

        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """

    def __init__(self, K):

        super().__init__()
        assert K > 0 # range has to be greater than 0
        self.K = K
        self.S = None # no GSO assigned yet
        self.N = None # no GSO assigned yet (N learned from the GSO)
        self.neighborhood = 'None' # no neighborhoods calculated yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(1,self.K+1))
        # Initialize parameters
        self.reset_parameters()
        
    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S
        # The neighborhood matrix has to be a tensor of shape
        #   nOutputNodes x maxNeighborhoodSize
        neighborhood = []
        maxNeighborhoodSizes = []
        for k in range(1,self.K+1):
            # For each hop (0,1,...) in the range K
            thisNeighborhood = graphTools.computeNeighborhood(
                            np.array(self.S), k, outputType='matrix')
            # compute the k-hop neighborhood
            neighborhood.append(torch.tensor(thisNeighborhood))
            maxNeighborhoodSizes.append(thisNeighborhood.shape[1])
        self.maxNeighborhoodSizes = maxNeighborhoodSizes
        self.neighborhood = neighborhood

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x N
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.N
        # And given that the self.neighborhood is already a torch.tensor matrix
        # we can just go ahead and get it.
        # So, x is of shape B x F x N. But we need it to be of shape
        # B x F x N x maxNeighbor. Why? Well, because we need to compute the
        # maximum between the value of each node and those of its neighbors.
        # And we do this by applying a torch.max across the rows (dim = 3) so
        # that we end up again with a B x F x N, but having computed the max.
        # How to fill those extra dimensions? Well, what we have is neighborhood
        # matrix, and we are going to use torch.gather to bring the right
        # values (torch.index_select, while more straightforward, only works
        # along a single dimension).
        # Each row of the matrix neighborhood determines all the neighbors of
        # each node: the first row contains all the neighbors of the first node,
        # etc.
        # The values of the signal at those nodes are contained in the dim = 2
        # of x. So, just for now, let's ignore the batch and feature dimensions
        # and imagine we have a column vector: N x 1. We have to pick some of
        # the elements of this vector and line them up alongside each row
        # so that then we can compute the maximum along these rows.
        # When we torch.gather along dimension 0, we are selecting which row to
        # pick according to each column. Thus, if we have that the first row
        # of the neighborhood matrix is [1, 2, 0] means that we want to pick
        # the value at row 1 of x, at row 2 of x in the next column, and at row
        # 0 of the last column. For these values to be the appropriate ones, we
        # have to repeat x as columns to build our b x F x N x maxNeighbor
        # matrix.
        xK = x # xK is a tensor aggregating the 0-hop (x), 1-hop, ..., K-hop
        # max's it is initialized with the 0-hop neigh. (x itself)
        xK = xK.unsqueeze(3) # extra dimension added for concatenation ahead
        x = x.unsqueeze(3) # B x F x N x 1
        # And the neighbors that we need to gather are the same across the batch
        # and feature dimensions, so we need to repeat the matrix along those
        # dimensions
        for k in range(1,self.K+1):
            x_aux = x.repeat([1, 1, 1, self.maxNeighborhoodSizes[k-1]])
            gatherNeighbor = self.neighborhood[k-1].reshape(
                                                [1,
                                                 1,
                                                 self.N,
                                                 self.maxNeighborhoodSizes[k-1]]
                                                )
            gatherNeighbor = gatherNeighbor.repeat([batchSize, 
                                                    dimNodeSignals,
                                                    1,
                                                    1])
            # And finally we're in position of getting all the neighbors in line
            xNeighbors = torch.gather(x_aux, 2, gatherNeighbor.long())
            #   B x F x nOutput x maxNeighbor
            # Note that this gather function already reduces the dimension to
            # nOutputNodes.
            # And proceed to compute the maximum along this dimension
            v, _ = torch.max(xNeighbors, dim = 3)
            v = v.unsqueeze(3) # to concatenate with xK
            xK = torch.cat((xK,v),3)
        out = torch.matmul(xK,self.weight.unsqueeze(2))
        # multiply each k-hop max by corresponding weight
        out = out.reshape([batchSize,dimNodeSignals,self.N])
        return out
    
    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.K)
        self.weight.data.uniform_(-stdv, stdv)
    
    def extra_repr(self):
        if self.neighborhood is not None:
            reprString = "neighborhood stored"
        else:
            reprString = "NO neighborhood stored"
        return reprString
    
class MedianLocalActivation(nn.Module):
    # Luana R. Ruiz, rubruiz@seas.upenn.edu, 2019/03/27
    """
    MedianLocalActivation creates a localized activation function layer on 
    graphs

    Initialization:

        MedianLocalActivation(K)

        Inputs:
            K (int): number of hops (>0)

        Output:
            torch.nn.Module for a localized median activation function layer

    Add graph shift operator:

        MedianLocalActivation.addGSO(GSO) Before applying the filter, we need 
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number 
        of edge features is the same; but the number of nodes can change).
        This function also calculates the 0-,1-,...,K-hop neighborhoods of every
        node

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = MedianLocalActivation(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes

        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """

    def __init__(self, K):

        super().__init__()
        assert K > 0 # range has to be greater than 0
        self.K = K
        self.S = None # no GSO assigned yet
        self.N = None # no GSO assigned yet (N learned from the GSO)
        self.neighborhood = 'None' # no neighborhoods calculated yet
        self.masks = 'None' # no mask yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(1,self.K+1))
        # Initialize parameters
        self.reset_parameters()
        
    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S
        # The neighborhood matrix has to be a tensor of shape
        #   nOutputNodes x maxNeighborhoodSize
        neighborhood = []
        for k in range(1,self.K+1):
            # For each hop (0,1,...) in the range K
            thisNeighborhood = graphTools.computeNeighborhood(
                            np.array(self.S), k, outputType='list')
            # compute the k-hop neighborhood
            neighborhood.append(thisNeighborhood)
        self.neighborhood = neighborhood

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x N
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.N
        xK = x # xK is a tensor aggregating the 0-hop (x), 1-hop, ..., K-hop
        # max's
        # It is initialized with the 0-hop neigh. (x itself)
        xK = xK.unsqueeze(3) # extra dimension added for concatenation ahead
        #x = x.unsqueeze(3) # B x F x N x 1
        for k in range(1,self.K+1):
            kHopNeighborhood = self.neighborhood[k-1] 
            # Fetching k-hop neighborhoods of all nodes
            kHopMedian = torch.empty(0)
            # Initializing the vector that will contain the k-hop median for
            # every node
            for n in range(self.N):
                # Iterating over the nodes
                # This step is necessary because here the neighborhoods are
                # lists of lists. It is impossible to pad them and feed them as
                # a matrix, as this would impact the outcome of the median
                # operation
                nodeNeighborhood = torch.tensor(np.array(kHopNeighborhood[n]))
                neighborhoodLen = len(nodeNeighborhood)
                gatherNode = nodeNeighborhood.reshape([1, 1, neighborhoodLen])
                gatherNode = gatherNode.repeat([batchSize, dimNodeSignals, 1])
                # Reshaping the node neighborhood for the gather operation
                xNodeNeighbors = torch.gather(x, 2, gatherNode.long())
                # Gathering signal values in the node neighborhood
                nodeMedian,_ = torch.median(xNodeNeighbors, dim = 2,
                                            keepdim=True)
                # Computing the median in the neighborhood
                kHopMedian = torch.cat([kHopMedian,nodeMedian],2)
                # Concatenating k-hop medians node by node
            kHopMedian = kHopMedian.unsqueeze(3) # Extra dimension for
            # concatenation with the previous (k-1)-hop median tensor 
            xK = torch.cat([xK,kHopMedian],3)
        out = torch.matmul(xK,self.weight.unsqueeze(2))
        # Multiplying each k-hop median by corresponding trainable weight
        out = out.reshape([batchSize,dimNodeSignals,self.N])
        return out
    
    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.K)
        self.weight.data.uniform_(-stdv, stdv)
    
    def extra_repr(self):
        if self.neighborhood is not None:
            reprString = "neighborhood stored"
        else:
            reprString = "NO neighborhood stored"
        return reprString
        
class NoActivation(nn.Module):
    """
    NoActivation creates an activation layer that does nothing
        It is for completeness, to be able to switch between linear models
        and nonlinear models, without altering the entire architecture model
    Initialization:
        NoActivation()
        Output:
            torch.nn.Module for an empty activation layer
    Forward call:
        y = NoActivation(x)
        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes
        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        
        return x
    
    def extra_repr(self):
        reprString = "No Activation Function"
        return reprString

class NoPool(nn.Module):
    """
    This is a pooling layer that actually does no pooling. It has the same input
    structure and methods of MaxPoolLocal() for consistency. Basically, this
    allows us to change from pooling to no pooling without necessarily creating
    a new architecture.
    
    In any case, we're pretty sure this function should never ship, and pooling
    can be avoided directly when defining the architecture.
    """

    def __init__(self, nInputNodes, nOutputNodes, nHops):

        super().__init__()
        self.nInputNodes = nInputNodes
        self.nOutputNodes = nOutputNodes
        self.nHops = nHops
        self.neighborhood = None

    def addGSO(self, GSO):
        # This is necessary to keep the form of the other pooling strategies
        # within the SelectionGNN framework. But we do not care about any GSO.
        pass

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x nInputNodes
        assert x.shape[2] == self.nInputNodes
        # Check that there are at least the same number of nodes that
        # we will keep (otherwise, it would be unpooling, instead of
        # pooling)
        assert x.shape[2] >= self.nOutputNodes
        # And do not do anything
        return x

    def extra_repr(self):
        reprString = "in_dim=%d, out_dim=%d, number_hops = %d, " % (
                self.nInputNodes, self.nOutputNodes, self.nHops)
        reprString += "no neighborhood needed"
        return reprString

class MaxPoolLocal(nn.Module):
    """
    MaxPoolLocal Creates a pooling layer on graphs by selecting nodes

    Initialization:

        MaxPoolLocal(in_dim, out_dim, number_hops)

        Inputs:
            in_dim (int): number of nodes at the input
            out_dim (int): number of nodes at the output
            number_hops (int): number of hops to pool information

        Output:
            torch.nn.Module for a local max-pooling layer.

        Observation: The selected nodes for the output are always the top ones.

    Add a neighborhood set:
        
    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before being used, we need to define the GSO 
        that will determine the neighborhood that we are going to pool.

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        v = MaxPoolLocal(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x in_dim

        Outputs:
            y (torch.tensor): pooled data; shape:
                batch_size x dim_features x out_dim
    """

    def __init__(self, nInputNodes, nOutputNodes, nHops):

        super().__init__()
        self.nInputNodes = nInputNodes
        self.nOutputNodes = nOutputNodes
        self.nHops = nHops
        self.neighborhood = None

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N (And I don't care about E, because the
        # computeNeighborhood function takes care of it)
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        # Get the device (before operating with S and losing it, it's cheaper
        # to store the device now, than to duplicate S -i.e. keep a numpy and a
        # tensor copy of S)
        device = S.device
        # Move the GSO to cpu and to np.array so it can be handled by the
        # computeNeighborhood function
        S = np.array(S.cpu())
        # Compute neighborhood
        neighborhood = graphTools.computeNeighborhood(S, self.nHops,
                                                      self.nOutputNodes,
                                                      self.nInputNodes,'matrix')
        # And move the neighborhood back to a tensor
        neighborhood = torch.tensor(neighborhood).to(device)
        # The neighborhood matrix has to be a tensor of shape
        #   nOutputNodes x maxNeighborhoodSize
        assert neighborhood.shape[0] == self.nOutputNodes
        assert neighborhood.max() <= self.nInputNodes
        # Store all the relevant information
        self.maxNeighborhoodSize = neighborhood.shape[1]
        self.neighborhood = neighborhood

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x nInputNodes
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.nInputNodes
        # Check that there are at least the same number of nodes that
        # we will keep (otherwise, it would be unpooling, instead of
        # pooling)
        assert x.shape[2] >= self.nOutputNodes
        # And given that the self.neighborhood is already a torch.tensor matrix
        # we can just go ahead and get it.
        # So, x is of shape B x F x N. But we need it to be of shape
        # B x F x N x maxNeighbor. Why? Well, because we need to compute the
        # maximum between the value of each node and those of its neighbors.
        # And we do this by applying a torch.max across the rows (dim = 3) so
        # that we end up again with a B x F x N, but having computed the max.
        # How to fill those extra dimensions? Well, what we have is neighborhood
        # matrix, and we are going to use torch.gather to bring the right
        # values (torch.index_select, while more straightforward, only works
        # along a single dimension).
        # Each row of the matrix neighborhood determines all the neighbors of
        # each node: the first row contains all the neighbors of the first node,
        # etc.
        # The values of the signal at those nodes are contained in the dim = 2
        # of x. So, just for now, let's ignore the batch and feature dimensions
        # and imagine we have a column vector: N x 1. We have to pick some of
        # the elements of this vector and line them up alongside each row
        # so that then we can compute the maximum along these rows.
        # When we torch.gather along dimension 0, we are selecting which row to
        # pick according to each column. Thus, if we have that the first row
        # of the neighborhood matrix is [1, 2, 0] means that we want to pick
        # the value at row 1 of x, at row 2 of x in the next column, and at row
        # 0 of the last column. For these values to be the appropriate ones, we
        # have to repeat x as columns to build our b x F x N x maxNeighbor
        # matrix.
        x = x.unsqueeze(3) # B x F x N x 1
        x = x.repeat([1, 1, 1, self.maxNeighborhoodSize]) # BxFxNxmaxNeighbor
        # And the neighbors that we need to gather are the same across the batch
        # and feature dimensions, so we need to repeat the matrix along those
        # dimensions
        gatherNeighbor = self.neighborhood.reshape([1, 1,
                                                    self.nOutputNodes,
                                                    self.maxNeighborhoodSize])
        gatherNeighbor = gatherNeighbor.repeat([batchSize, dimNodeSignals, 1,1])
        # And finally we're in position of getting all the neighbors in line
        xNeighbors = torch.gather(x, 2, gatherNeighbor)
        #   B x F x nOutput x maxNeighbor
        # Note that this gather function already reduces the dimension to
        # nOutputNodes.
        # And proceed to compute the maximum along this dimension
        v, _ = torch.max(xNeighbors, dim = 3)
        return v

    def extra_repr(self):
        reprString = "in_dim=%d, out_dim=%d, number_hops = %d, " % (
                self.nInputNodes, self.nOutputNodes, self.nHops)
        if self.neighborhood is not None:
            reprString += "neighborhood stored"
        else:
            reprString += "NO neighborhood stored"
        return reprString

class GraphFilter(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E = 1, bias = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        u = LSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString

class SpectralGF(nn.Module):
    """
    SpectralGF Creates a (linear) layer that applies a LSI graph filter in the
        spectral domain using a cubic spline if needed.

    Initialization:

        GraphFilter(in_features, out_features, filter_coeff,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_coeff (int): number of filter spectral coefficients
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer) implemented in the spectral domain.

        Observation: Filter taps have shape
            out_features x edge_features x in_features x filter_coeff

    Add graph shift operator:

        SpectralGF.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = SpectralGF(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, M, E = 1, bias = True):
        # GSOs will be added later.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.M = M
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, G, M))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.M)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has to have 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S # Save S
        # Now we need to compute the eigendecomposition and save it
        # To compute the eigendecomposition, we use numpy.
        # So, first, get S in numpy format.
        Snp = np.array(S.data.cpu())
        # We will compute the eigendecomposition for each edge feature, so we
        # create the E x N x N space for V, VH and Lambda (we need lambda for
        # the spline kernel)
        V = np.zeros([self.E, self.N, self.N])
        VH = np.zeros([self.E, self.N, self.N])
        Lambda = np.zeros([self.E, self.N])
        # Here we save the resulting spline kernel matrix
        splineKernel = np.zeros([self.E, self.N, self.M])
        for e in range(self.E):
            # Compute the eigendecomposition
            Lambda[e,:], V[e,:,:] = np.linalg.eig(Snp[e,:,:])
            # Compute the hermitian
            VH[e,:,:] = V[e,:,:].conj().T
            # Compute the splineKernel basis matrix
            splineKernel[e,:,:] = graphTools.splineBasis(self.M, Lambda[e,:])
        # Transform everything to tensors of appropriate type on appropriate
        # device, and store them.
        self.V = torch.tensor(V).type(S.dtype).to(S.device) # E x N x N
        self.VH = torch.tensor(VH).type(S.dtype).to(S.device) # E x N x N
        self.splineKernel = torch.tensor(splineKernel)\
                                .type(S.dtype).to(S.device)
            # E x N x M
        # Once we have computed the splineKernel, we do not need to save the
        # eigenvalues.

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]

        # Check if we have enough spectral filter coefficients as needed, or if
        # we need to fill out the rest using the spline kernel.
        if self.M == self.N:
            self.h = self.weight # F x E x G x N (because N = M)
        else:
            # Adjust dimensions for proper algebraic matrix multiplication
            splineKernel = self.splineKernel.reshape([1,self.E,self.N,self.M])
            # We will multiply a 1 x E x N x M matrix with a F x E x M x G
            # matrix to get the proper F x E x N x G coefficients
            self.h = torch.matmul(splineKernel, self.weight.permute(0,1,3,2))
            # And now we rearrange it to the same shape that the function takes
            self.h = self.h.permute(0,1,3,2) # F x E x G x N
        # And now we add the zero padding (if this comes from a pooling
        # operation)
        if Nin < self.N:
            zeroPad = torch.zeros(B, F, self.N-Nin).type(x.dtype).to(x.device)
            x = torch.cat((x, zeroPad), dim = 2)
        # Compute the filter output
        u = spectralGF(self.h, self.V, self.VH, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString

class NodeVariantGF(nn.Module):
    """
    NodeVariantGF Creates a filtering layer that applies a node-variant graph
        filter

    Initialization:

        NodeVariantGF(in_features, out_features, shift_taps, node_taps
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            shift_taps (int): number of filter taps for shifts
            node_taps (int): number of filter taps for nodes
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer using node-variant graph
                filters.

        Observation: Filter taps have shape
            out_features x edge_features x shift_taps x in_features x node_taps

    Add graph shift operator:

        NodeVariantGF.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = NodeVariantGF(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, M, E = 1, bias = True):
        # G: Number of input features
        # F: Number of output features
        # K: Number of filter shift taps
        # M: Number of filter node taps
        # GSOs will be added later.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.M = M
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G, M))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K * self.M)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S
        npS = np.array(S.data.cpu()) # Save the GSO as a numpy array because we
        # are going to compute the neighbors.
        # And now we have to fill up the parameter vector, from M to N
        if self.M < self.N:
            # The first elements of M (ordered with whatever order we want)
            # are the ones associated to independent node taps.
            copyNodes = [m for m in range(self.M)]
            # The rest of the nodes will copy one of these M node taps.
            # The way we do this is: if they are connected to one of the M
            # indepdendent nodes, just copy it. If they are not connected,
            # look at the neighbors, neighbors, and so on, until we reach one
            # of the independent nodes.
            # Ties are broken by selecting the node with the smallest index
            # (which, due to the ordering, is the most important node of all
            # the available ones)
            neighborList = graphTools.computeNeighborhood(npS, 1,
                                                          nb = self.M)
            # This gets the list of 1-hop neighbors for all nodes.
            # Find the nodes that have no neighbors
            nodesWithNoNeighbors = [n for n in range(self.N) \
                                                   if len(neighborList[n]) == 0]
            # If there are still nodes that didn't find a neighbor
            K = 1 # K-hop neighbor we have looked so far
            while len(nodesWithNoNeighbors) > 0:
                # Looks for the next hop
                K += 1
                # Get the neigbors one further hop away
                thisNeighborList = graphTools.computeNeighborhood(npS,
                                                                  K,
                                                                  nb = self.M)
                # Check if we now have neighbors for those that didn't have
                # before
                for n in nodesWithNoNeighbors:
                    # Get the neighbors of the node
                    thisNodeList = thisNeighborList[n]
                    # If there are neighbors
                    if len(thisNodeList) > 0:
                        # Add them to the list
                        neighborList[n] = thisNodeList
                # Recheck if all nodes have non-empty neighbors
                nodesWithNoNeighbors = [n for n in range(self.N) \
                                                   if len(neighborList[n]) == 0]
            # Now we have obtained the list of independent nodes connected to
            # all nodes, we keep the one with highest score. And since the
            # matrix is already properly ordered, this means keeping the
            # smallest index in the neighborList.
            for m in range(self.M, self.N):
                copyNodes.append(min(neighborList[m]))
            # And, finally create the indices of nodes to copy
            self.copyNodes = torch.tensor(copyNodes).to(S.device)
        elif self.M == self.N:
            # In this case, all parameters go into the vector h
            self.copyNodes = torch.arange(self.M).to(S.device)
        else:
            # This is the rare case in which self.M < self.N, for example, if
            # we train in a larger network and deploy in a smaller one. Since
            # the matrix is ordered by score, we just keep the first N
            # weights
            self.copyNodes = torch.arange(self.N).to(S.device)
        # OBS.: self.weight is updated on each training step, so we cannot
        # define the self.h vector (i.e. the vector with N elements) here,
        # because otherwise it wouldn't be updated every time. So we need, in
        # the for, to use index_select on the actual weights, to create the
        # vector h that is later feed into the NVGF computation.

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # If we have less filter coefficients than the required ones, we need
        # to use the copying scheme
        if self.M == self.N:
            self.h = self.weight
        else:
            self.h = torch.index_select(self.weight, 4, self.copyNodes)
        # And now we add the zero padding
        if Nin < self.N:
            zeroPad = torch.zeros(B, F, self.N-Nin).type(x.dtype).to(x.device)
            x = torch.cat((x, zeroPad), dim = 2)
        # Compute the filter output
        u = NVGF(self.h, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "shift_taps=%d, node_taps=%d, " % (
                        self.K, self.M) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString

class EdgeVariantGF(nn.Module):
    """
    EdgeVariantGF Creates a (linear) layer that applies an edge-variant graph
        filter using the masking approach. If less nodes than the total number 
        of nodes are selected, then the remaining nodes adopt an LSI filter
        (i.e. it becomes a hybrid edge-variant grpah filter)

    Initialization:

        EdgeVariantGF(in_features, out_features, shift_taps,
                      selected_nodes, number_nodes,
                      edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            shift_taps (int): number of shifts to consider
            selected_nodes (int): number of selected nodes to implement the EV
                part of the filter
            number_nodes (int): number of nodes
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer using hybrid
                edge-variant graph filters.

        Observation: Filter taps have shape
            out_features x edge_features x shift_taps x in_features
                x number_nodes x number_nodes
            These weights are masked by the corresponding sparsity pattern of
            the graph and the desired number of selected nodes, so only weights
            in the nonzero edges of these nodes will be trained, the
            rest of the parameters contain trash. Therefore, the number of
            parameters will not reflect the actual number of parameters being
            trained.

    Add graph shift operator:

        EdgeVariantGF.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = EdgeVariantGF(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """
    def __init__(self, G, F, K, M, N, E=1, bias = True):
        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.M = M # Number of selected nodes
        self.N = N # Total number of nodes
        self.S = None
        # Create parameters for the Edge-Variant part:
        self.weightEV = nn.parameter.Parameter(torch.Tensor(F, E, K, G, N, N))
        # If we want a hybrid, create parameters
        if self.M < self.N:
            self.weightLSI = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        else:
            self.register_parameter('weightLSI', None)
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K * self.N)
        self.weightEV.data.uniform_(-stdv, stdv)
        if self.weightLSI is not None:
            self.weightLSI.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S # Save the GSO
        # Get the identity matrix across all edge features
        multipleIdentity = torch.eye(self.N).reshape([1, self.N, self.N])\
                            .repeat(self.E, 1, 1).to(S.device)
        # Compute the nonzero elements of S+I_{N}
        sparsityPattern = ((torch.abs(S) + multipleIdentity) > zeroTolerance)
        # Change from byte tensors to float tensors (or the same type of data as
        # the GSO)
        sparsityPattern = sparsityPattern.type(S.dtype)
        # But now we need to kill everything that is between elements M and N
        # (only if M < N)
        if self.M < self.N:
            # Create the ones in the row
            hybridMaskOnesRows = torch.ones([self.M, self.N])
            # Create the ones int he columns
            hybridMaskOnesCols = torch.ones([self.N - self.M, self.M])
            # Create the zeros
            hybridMaskZeros = torch.zeros([self.N - self.M, self.N - self.M])
            # Concatenate the columns
            hybridMask = torch.cat((hybridMaskOnesCols,hybridMaskZeros), dim=1)
            # Concatenate the rows
            hybridMask = torch.cat((hybridMaskOnesRows,hybridMask), dim=0)
        else:
            hybridMask = torch.ones([self.N, self.N])
        # Now that we have the hybrid mask, we need to mask the sparsityPattern
        # we got so far
        hybridMask = hybridMask.reshape([1, self.N, self.N]).to(S.device)
        #   1 x N x N
        sparsityPattern = sparsityPattern * hybridMask
        self.sparsityPattern = sparsityPattern.to(S.device)
        #   E x N x N
        # This gives the sparsity pattern for each edge feature
        # Now, let's create it of the right shape, so we do not have to go
        # around wasting time with reshapes when called in the forward
        # The weights have shape F x E x K x G x N x N
        # The sparsity pattern has shape E x N x N. And we want to make it
        # 1 x E x K x 1 x N x N. The K dimension is to guarantee that for k=0
        # we have the identity
        multipleIdentity = (multipleIdentity * hybridMask)\
                                    .reshape([1, self.E, 1, 1, self.N, self.N])
        # This gives a 1 x E x 1 x 1 x N x N identity matrix
        sparsityPattern = sparsityPattern\
                                    .reshape([1, self.E, 1, 1, self.N, self.N])
        # This gives a 1 x E x 1 x 1 x N x N sparsity pattern matrix
        sparsityPattern = sparsityPattern.repeat(1, 1, self.K-1, 1, 1, 1)
        # This repeats the sparsity pattern K-1 times giving a matrix of shape
        #   1 x E x (K-1) x 1 x N x N
        sparsityPattern = torch.cat((multipleIdentity,sparsityPattern), dim = 2)
        # This sholud give me a 1 x E x K x 1 x N x N matrix with the identity
        # in the first element
        self.sparsityPatternFull = sparsityPattern.type(S.dtype).to(S.device)

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # Mask the parameters
        self.Phi = self.weightEV * self.sparsityPatternFull
        # And now we add the zero padding
        if Nin < self.N:
            zeroPad = torch.zeros(B, F, self.N-Nin).type(x.dtype).to(x.device)
            x = torch.cat((x, zeroPad), dim = 2)
        # Compute the filter output for the EV part
        uEV = EVGF(self.Phi, x, self.bias)
        # Check if we need an LSI part
        if self.M < self.N:
            # Compute the filter output for the LSI part
            uLSI = LSIGF(self.weightLSI, self.S, x, self.bias)
        else:
            # If we don't, just add zero
            uLSI = torch.tensor(0., dtype = uEV.dtype).to(uEV.device)
        # Add both
        u = uEV + uLSI
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "shift_taps=%d, " % (
                        self.K) + \
                        "selected_nodes=%d, " % (self.M) +\
                        "number_nodes=%d, " % (self.N) +\
                        "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString

class GraphAttentional(nn.Module):
    """
    GraphAttentional Creates a graph attentional layer

    Initialization:

        GraphAttentional(in_features, out_features, attention_heads,
                         edge_features=1, nonlinearity=nn.functional.relu,
                         concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for a graph attentional layer.

    Add graph shift operator:

        GraphAttentional.addGSO(GSO) Before applying the filter, we need to
        define the GSO that we are going to use. This allows to change the GSO
        while using the same filtering coefficients (as long as the number of
        edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E = 1,
        nonlinearity = nn.functional.relu, concatenate = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(K, E, 2*F))
        self.weight = nn.parameter.Parameter(torch.Tensor(K, E, F, G))
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        self.mixer.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # And get the graph attention output
        y = graphAttention(x, self.mixer, self.weight, self.S)
        # This output is of size B x K x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the nonlinearity
            y = self.nonlinearity(y)
            # Concatenate: Make it B x KF x N such that first iterates over f
            # and then over k: (k=0,f=0), (k=0,f=1), ..., (k=0,f=F-1), (k=1,f=0),
            # (k=1,f=1), ..., etc.
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.K*self.F])\
                    .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim = 1) # B x F x N
            # And then we apply the nonlinearity
            y = self.nonlinearity(y)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "attention_heads=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString