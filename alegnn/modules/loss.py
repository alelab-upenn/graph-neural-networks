# 2021/03/04~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
loss.py Loss functions

adaptExtraDimensionLoss: wrapper that handles extra dimensions
F1Score: loss function corresponding to 1 - F1 score
"""

import torch
import torch.nn as nn

# An arbitrary loss function handling penalties needs to have the following
# conditions
# .penaltyList attribute listing the names of the penalties
# .nPenalties attibute is an int with the number of penalties
# Forward function has to output the actual loss, the main loss (with no
# penalties), and a dictionary with the value of each of the penalties.
# This will be standard procedure for all loss functions that have penalties.
# Note: The existence of a penalty will be signaled by an attribute in the model

class adaptExtraDimensionLoss(nn.modules.loss._Loss):
    """
    adaptExtraDimensionLoss: wrapper that handles extra dimensions
    
    Some loss functions take vectors as inputs while others take scalars; if we
    input a one-dimensional vector instead of a scalar, although virtually the
    same, the loss function could complain.
    
    The output of the GNNs is, by default, a vector. And sometimes we want it
    to still be a vector (i.e. crossEntropyLoss where we output a one-hot 
    vector) and sometimes we want it to be treated as a scalar (i.e. MSELoss).
    Since we still have a single training function to train multiple models, we
    do not know whether we will have a scalar or a vector. So this wrapper
    adapts the input to the loss function seamlessly.
    
    Eventually, more loss functions could be added to the code below to better
    handle their dimensions.
    
    Initialization:
        
        Input:
            lossFunction (torch.nn loss function): desired loss function
            arguments: arguments required to initialize the loss function
            >> Obs.: The loss function gets initialized as well
            
    Forward:
        Input:
            estimate (torch.tensor): output of the GNN
            target (torch.tensor): target representation
    """
    
    # When we want to compare scalars, we will have a B x 1 output of the GNN,
    # since the number of features is always there. However, most of the scalar
    # comparative functions take just a B vector, so we have an extra 1 dim
    # that raises a warning. This container will simply get rid of it.
    
    # This allows to change loss from crossEntropy (class based, expecting 
    # B x C input) to MSE or SmoothL1Loss (expecting B input)
    
    def __init__(self, lossFunction, *args):
        # The second argument is optional and it is if there are any extra 
        # arguments with which we want to initialize the loss
        
        super().__init__()
        
        if len(args) > 0:
            self.loss = lossFunction(*args) # Initialize loss function
        else:
            self.loss = lossFunction()
        
    def forward(self, estimate, target):
        
        # What we're doing here is checking what kind of loss it is and
        # what kind of reshape we have to do on the estimate
        
        if 'CrossEntropyLoss' in repr(self.loss):
            # This is supposed to be a one-hot vector batchSize x nClasses
            assert len(estimate.shape) == 2
        elif 'SmoothL1Loss' in repr(self.loss) \
                    or 'MSELoss' in repr(self.loss) \
                    or 'L1Loss' in repr(self.loss):
            # In this case, the estimate has to be a batchSize tensor, so if
            # it has two dimensions, the second dimension has to be 1
            if len(estimate.shape) == 2:
                assert estimate.shape[1] == 1
                estimate = estimate.squeeze(1)
            assert len(estimate.shape) == 1
        
        return self.loss(estimate, target)
    
def F1Score(yHat, y):
# Luana R. Ruiz, rubruiz@seas.upenn.edu, 2021/03/04
    dimensions = len(yHat.shape)
    C = yHat.shape[dimensions-2]
    N = yHat.shape[dimensions-1]
    yHat = yHat.reshape((-1,C,N))
    yHat = torch.nn.functional.log_softmax(yHat, dim=1)
    yHat = torch.exp(yHat)
    yHat = yHat[:,1,:]
    y = y.reshape((-1,N))
    
    tp = torch.sum(y*yHat,1)
    #tn = torch.sum((1-y)*(1-yHat),1)
    fp = torch.sum((1-y)*yHat,1)
    fn = torch.sum(y*(1-yHat),1)

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    
    idx_p = p!=p
    idx_tp = tp==0
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