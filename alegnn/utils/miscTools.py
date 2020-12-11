# 2018/10/15~
# Fernando Gama, fgama@seas.upenn.edu.
# Luana Ruiz, rubruiz@seas.upenn.edu.
"""
miscTools Miscellaneous Tools module

num2filename: change a numerical value into a string usable as a filename
saveSeed: save the random state of generators
loadSeed: load the number of random state of generators
writeVarValues: write the specified values in the specified txt file
"""

import os
import pickle
import numpy as np
import torch

def num2filename(x,d):
    """
    Takes a number and returns a string with the value of the number, but in a
    format that is writable into a filename.

    s = num2filename(x,d) Gets rid of decimal points which are usually
        inconvenient to have in a filename.
        If the number x is an integer, then s = str(int(x)).
        If the number x is a decimal number, then it replaces the '.' by the
        character specified by d. Setting d = '' erases the decimal point,
        setting d = '.' simply returns a string with the exact same number.

    Example:
        >> num2filename(2,'d')
        >> '2'

        >> num2filename(3.1415,'d')
        >> '3d1415'

        >> num2filename(3.1415,'')
        >> '31415'

        >> num2filename(3.1415,'.')
        >> '3.1415'
    """
    if x == int(x):
        return str(int(x))
    else:
        return str(x).replace('.',d)

def saveSeed(randomStates, saveDir):
    """
    Takes a list of dictionaries of random generator states of different modules
    and saves them in a .pkl format.
    
    Inputs:
        randomStates (list): The length of this list is equal to the number of
            modules whose states want to be saved (torch, numpy, etc.). Each
            element in this list is a dictionary. The dictionary has three keys:
            'module' with the name of the module in string format ('numpy' or
            'torch', for example), 'state' with the saved generator state and,
            if corresponds, 'seed' with the specific seed for the generator
            (note that torch has both state and seed, but numpy only has state)
        saveDir (path): where to save the seed, it will be saved under the 
            filename 'randomSeedUsed.pkl'
    """
    pathToSeed = os.path.join(saveDir, 'randomSeedUsed.pkl')
    with open(pathToSeed, 'wb') as seedFile:
        pickle.dump({'randomStates': randomStates}, seedFile)
        
def loadSeed(loadDir):
    """
    Loads the states and seed saved in a specified path
    
    Inputs:
        loadDir (path): where to look for thee seed to load; it is expected that
            the appropriate file within loadDir is named 'randomSeedUsed.pkl'
    
    Obs.: The file 'randomSeedUsed.pkl' should contain a list structured as
        follows. The length of this list is equal to the number of modules whose
        states were saved (torch, numpy, etc.). Each element in this list is a
        dictionary. The dictionary has three keys: 'module' with the name of 
        the module in string format ('numpy' or 'torch', for example), 'state' 
        with the saved generator state and, if corresponds, 'seed' with the 
        specific seed for the generator (note that torch has both state and 
        seed, but numpy only has state)
    """
    pathToSeed = os.path.join(loadDir, 'randomSeedUsed.pkl')
    with open(pathToSeed, 'rb') as seedFile:
        randomStates = pickle.load(seedFile)
        randomStates = randomStates['randomStates']
    for module in randomStates:
        thisModule = module['module']
        if thisModule == 'numpy':
            np.random.RandomState().set_state(module['state'])
        elif thisModule == 'torch':
            torch.set_rng_state(module['state'])
            torch.manual_seed(module['seed'])
                

def writeVarValues(fileToWrite, varValues):
    """
    Write the value of several string variables specified by a dictionary into
    the designated .txt file.
    
    Input:
        fileToWrite (os.path): text file to save the specified variables
        varValues (dictionary): values to save in the text file. They are
            saved in the format "key = value".
    """
    with open(fileToWrite, 'a+') as file:
        for key in varValues.keys():
            file.write('%s = %s\n' % (key, varValues[key]))
        file.write('\n')
