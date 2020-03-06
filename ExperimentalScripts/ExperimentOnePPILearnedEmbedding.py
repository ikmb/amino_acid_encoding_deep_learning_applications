#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hesham El Abd
@brief: writing the code for Experiment one which studies the interaction
between the amount of data and the embedding dimensionality on model generalizability.  
"""
## load the modules
import os
## make a parent directory to store the variables:
os.makedirs(name="ExperimentOnePPI", exist_ok=True)
## construct the experimental variables: 
dimensions=[1,2,4,8,16,32]
dataFractions=[0.25,0.5,0.75,1.0]
numberOfReps=5
gpuIndex=6 # Execute the code on the first GPU 
trainedEmbeeding=True # The Embeedng dimension is trainable.
for dimension in dimensions:
    os.makedirs(name="ExperimentOnePPIExtension/EmbeddingDimension"+str(dimension),
                exist_ok=True)
    for repIndex in range(numberOfReps):
        os.makedirs(name="ExperimentOnePPIExtension/EmbeddingDimension"+
                    str(dimension)+"/Rep"+str(repIndex+1),
                    exist_ok=True)
        for fraction in dataFractions:
            os.system("python trainDPPI2.py"+
                      " -n 21"+" "+
                      "-d "+str(dimension)+" "+
                      "-t "+str(int(trainedEmbeeding))+" "+
                      "-g "+str(gpuIndex)+" "+
                      "-f "+str(fraction)+" "+
                      "-o "+
                      "ExperimentOnePPIExtension/EmbeddingDimension"+
                      str(dimension)+"/Rep"+str(repIndex+1)+
                      "/ModelPerformanceOn_"+
                      str(fraction)+"%_OfTrainingData")
print("Experiment One has been Executed")