#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hesham ElAbd
@group Genetics and Bioinformatics group, IKMB, University of Kiel, Germany
@contact h.elabd@ikmb.uni-kiel.de
@brief A command-line script for constructing and training an LSTM based peptide-HLA-II interaction model        
@see RNNBluePrint.py  
@note: The model uses One-Hot encoding to encode the provided amino acids
"""
# load the modules:
import tensorflow as tf
import numpy as np 
import pandas as pd
import pickle 
from RNNBluePrint import buildHlaCNN_RNNModelOneHot
import argparse
import os
from sklearn.metrics import roc_auc_score
# parse the user inputs:
parser=argparse.ArgumentParser()

parser.add_argument('-g',
                    '--gpuIndex',
                    help="""The index of the GPU incase training is being carried 
                             on a multi-GPU machine""",
                    type=int)

parser.add_argument('-o',
                    '--outputPath',
                    help=""" The output path to save the results of the model
                    """,
                    type=str)



parser.add_argument('-i','--inputPath',
                    help=""" The path to load the input training data""",
                    type=str)

parser.add_argument('-v','--valPath',
                    help=""" The path to load the validation dataset""",
                    type=str)
## parsing the user inputs
inputs=parser.parse_args()
gpuIndex=inputs.gpuIndex
outputPath=inputs.outputPath
trainDataPath=inputs.inputPath
testDataPath=inputs.valPath
# restrict  memory growth: 
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuIndex)
for gpu in tf.config.experimental.get_visible_devices()[1:]:
    tf.config.experimental.set_memory_growth(gpu,True)
## load and Encode the train  dataset: 
trainData=pd.read_csv(trainDataPath)
trainlabels=np.array(trainData.iloc[:,1].tolist())
trainTensor=tf.keras.preprocessing.sequence.pad_sequences(sequences=
            tokenizer.texts_to_sequences(trainData.iloc[:,0]),dtype=np.int32,
            maxlen=37,padding="pre")
## load and Encode the test  dataset:
testData=pd.read_csv(testDataPath)
testlabels=np.array(testData.iloc[:,1].tolist())
testTensor=tf.keras.preprocessing.sequence.pad_sequences(sequences=
            tokenizer.texts_to_sequences(testData.iloc[:,0]),dtype=np.int32,
            maxlen=37,padding="pre")
## assert that loading and encoding was correct
assert  tf.reduce_sum(trainTensor) != tf.reduce_sum(testTensor), """ something went wrong with loading and encoding the 
data but train and test tensor should not be identical, please check your input"""
## encode the tensors using One-Hot encoding: 
trainTensor=tf.one_hot(trainTensor, depth=21)
testTensor=tf.one_hot(testTensor, depth=21)
# check the correctness of the output:
assert len(trainTensor.shape)==3, """ something went wrong, I was expecting a 
rank 3 tensor, however, trainTensor has a ranke {}""".format(
len(trainTensor.shape))
assert len(testTensor.shape)==3, """ something went wrong, I was expecting a 
rank 3 tensor, however, testTensor has a ranke {}""".format(
len(testTensor.shape))

## create the model
model=buildHlaCNN_RNNModelOneHot(inputSize=37,numberOfTokens=21)
## define a function to compute the AUC: 
def dynamicAUC(y_true,y_pred,threshold=0.426):
    """
    @brief a wrapper function for computing the AUC
    @detailsThe function provided an interface to compute the AUC inside a TensorFlow
                computational graph. It, first binarizes its the experimentally measured
                labels, i.e. y_true, using the threshold used with netMHCpan 3.2, finally 
                it uses `py_function` to wrap the `roc_auc_score` function and uses the 
                predicted labels and true labels to compute the  AUC.
    @param y_true: tensor, a rank one tensor, which is the experimentally measured values.
    @param y_pred: tensor, a rank one tensor, the predicted affinity of the model. 
    @param threshold: scaler, a float, threshold for binarization.
    """
    casetedLabels=tf.cast(y_true>=0.426,tf.int32)
    return tf.py_function(roc_auc_score, (casetedLabels, y_pred), 
                          tf.double,name='prediction')
    
## compile the model: 
model.compile(loss="mse",optimizer="adam",metrics=["mae",dynamicAUC])
## define a callback to train the model: 
if outputPath[-1] != "/":
    outputPath+="/"
csvCallBack=tf.keras.callbacks.CSVLogger(
        outputPath+"model_OneHot.csv")

model.fit(x=trainTensor,y=trainlabels,
          validation_data=(testTensor,testlabels),epochs=3000,batch_size=256,
          callbacks=[csvCallBack])




















