#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:38:54 2019
@author: Hesham El Abd
@brief: writing the code for Experiment one which studies the interaction
between the amount of data and VHSE8 encoding on the model generalizability.  
"""
## load the modules
import os
## make a parent directory to store the variables:
os.makedirs(name="ExperimentFourPPI", exist_ok=True)
## construct the experimental variables: 
dataFractions=[0.25,0.5,0.75,1.0]
numberOfReps=5
gpuIndex=0 # Execute the code on the first GPU 
## construct a training loop:
for repIndex in range(numberOfReps):
    os.makedirs(name="ExperimentFourPPI/"+"/Rep"+str(repIndex+1),
                exist_ok=True)
    for fraction in dataFractions:
        os.system("python trainDPPIModelOnVHSE8.py"+
                      " "+
                      "-g "+str(gpuIndex)+" "+
                      "-f "+str(fraction)+" "+
                      "-o "+
                      "ExperimentFourPPI/Rep"+str(repIndex+1)+
                      "/ModelPerformanceOn_"+
                      str(fraction)+"%_OfTrainingData")
print("Experiment Three has been Executed")
