import re
import numpy as np
import pandas as pd 
## define the functions: 
# define a function to split the input sequences
def splitSequenceintoKmers(listOfSequences,k):
    pattern="".join([""+"." for _ in range(k)])+"?"
    fragmentedSequences=[]
    for protein in listOfSequences: 
        fragmentedProteinList=re.findall(pattern,protein)
        dumString=""
        for twoMers in fragmentedProteinList:
            dumString+=twoMers+" "
        fragmentedSequences.append(dumString.strip(" "))
    return fragmentedSequences

# define a shuffiling function: 
def shuffleArray(a, b, c):
    """
    A modified version of the unison_shuffled_copies function proposed by 
    Íhor Mé @https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    """
    assert a.shape[0] == b.shape[0]
    assert c.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p], c[p]
