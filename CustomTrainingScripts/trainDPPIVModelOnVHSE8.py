#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hesham ElAbd
@group Genetics and Bioinformatics group, IKMB, University of Kiel, Germany
@contact h.elabd@ikmb.uni-kiel.de

@brief A command-line script for constructing and training a DPPI model. 
@details Constructing a DPPI models with a user defined parameters, train it on a specific fraction of input data 
        using a specific GPU and then write the results dict as pickle object on a user defined object.
        The loss, metrics and model checkpoints are written a log directory which enable live monitoring of 
        the training using TensorBoard. 
@see DPPIBluePrint.py
@note: The model uses VHSE8 Matrix to encode the provided amino acids
"""
# load the modules:
import tensorflow as tf
import pickle
import os
from DPPIBluePrint import DPPIModel
import argparse
import pandas as pd
import numpy as np
## define the user input:
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

parser.add_argument('-f',
                    '--dataFrac',
                    help=""" The Fraction of Traing data to train the models on. 
                           A float that is bigger than zero and smaller than or 
                           equal to one.""",
                    type=float)
## Parsing the input parameters: 
inputs=parser.parse_args()
gpuIndex=inputs.gpuIndex
outputPath=inputs.outputPath
dataFrac=inputs.dataFrac
## Load the tokenizer: 
with open("OneMerAminoAcidTokenizer.pickle","rb") as input_:
    tokenizer=pickle.load(input_)
## load the VHSE8 matrix
raw_VHSE=pd.read_csv('VHSE8.csv')
# prepare VHSE for embedding: 
VHSE=np.zeros((21,8),dtype=np.float32)
count=0
for index, char in enumerate(tokenizer.word_index):
    count+=1
    embedding_vector= np.array(raw_VHSE.loc[raw_VHSE.AA==char.upper()])[:,1:9].reshape(8,)
    VHSE[index+1,:]=[embedding_vector[i].replace("−",'-') for i in range(embedding_vector.shape[0])]
    if count ==20: # breake after getting the index of the 20th amino acid as VHSE8 only encode the standerd 20 amino acids
        break
# validate the user input for data fraction
assert dataFrac >=0 and dataFrac<=1,""" data fraction should be a float between 
zero and one which specify The fraction of the training data to use.
Your input is: {} """.format(dataFrac)
# Adjust the memory utilization by GPUs:
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuIndex)
for gpu in tf.config.experimental.get_visible_devices()[1:]:
    tf.config.experimental.set_memory_growth(gpu,True)
## Read the data 
with open("OneKmerArrayOneTrainEqual.pickle","rb") as output_:
    firstProteinArrayTrain=pickle.load(output_)

with open("OneKmerArrayTwoTrainEqual.pickle","rb") as output_:
    secondProteinArrayTrain=pickle.load(output_)
    
with open("OneKmerArrayThreeTrainEqual.pickle","rb") as output_:
    thirdProteinsListTrain=pickle.load(output_)

with open("OneKmerArrayOneTestEqual.pickle","rb") as output_:
    firstProteinArrayTest=pickle.load(output_)

with open("OneKmerArrayTwoTestEqual.pickle","rb") as output_:
    secondProteinArrayTest=pickle.load(output_)

with open("OneKmerArrayThreeTestEqual.pickle","rb") as output_:
    thirdProteinsListTest=pickle.load(output_)
# define a model for training the data:
model=DPPIModel(21,8,False,VHSE) # set the embedding matrix and prevent the updates in the embedding matrix
## construct evaluation metrics for training and Test datasets: 
# Training metrics: 
trainLoss=tf.keras.metrics.Mean(name="TrainLoss")
trainAcuracy=tf.keras.metrics.Accuracy(name="TrainAccuracy")
trainAUC=tf.keras.metrics.AUC(name="TrainAUC")
trainRecall=tf.keras.metrics.Recall(name="TrainRecall")
trainPrecision=tf.keras.metrics.Precision(name="TrainPercision") 
# test metrics: 
testLoss=tf.keras.metrics.Mean(name="TestLoss")
testAcuracy=tf.keras.metrics.Accuracy(name="TestAccuracy")
testAUC=tf.keras.metrics.AUC(name="TestAUC")
testRecall=tf.keras.metrics.Recall(name="TestRecall")
testPrecision=tf.keras.metrics.Precision(name="TestPercision") 
## define the train and evaluation loop:
# define the input signature:
InputSignature=[
         tf.TensorSpec(shape=(None,firstProteinArrayTrain.shape[1]), 
                       dtype=tf.int32),
         tf.TensorSpec(shape=(None,firstProteinArrayTrain.shape[1]), 
                       dtype=tf.int32),
         tf.TensorSpec(shape=(None,1), dtype=tf.float64)
        ]
# declare the optimizer and the loss function.
optimizer=tf.keras.optimizers.Adam()
lossFunction=tf.keras.losses.BinaryCrossentropy()
# define the training step function 
@tf.function(input_signature=InputSignature)
def trainStep(inputATensor,inputBTensor,label):
    """
    @brief train the model on a batch of data 
    @details The function train the model using one batch of the input data. 
             Training is carried out using GradientTapes. 
    @param inputATensor: tensor, a rank two tensor of shape (batch_size, seq_len) which contain 
    the tokenized sequences of the first protein. 
    @param inputBTensor: tensor, a rank two tensor of shape (batch_size, seq_len) which contain 
    the tokenized sequences of the second protein. 
    @param label: tensor, a rank one tensor which store for each pair of input sequences whether
    they interact of not.
    @note: During Accuracy calculations the model predictions are casted 
    as binary values, i.e. if the predictions are 0.5 AND above they are
    casted to One otherwise to zero.
    """
    with tf.GradientTape() as tape: 
        modelPrediction=model(inputATensor,inputBTensor,True)
        loss=lossFunction(y_true=label,y_pred=modelPrediction)
    grads=tape.gradient(loss,model.trainable_variables) 
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    trainLoss(loss)
    trainAcuracy(y_true=label,y_pred=tf.cast(modelPrediction>=0.5,tf.int32))
    trainAUC(y_true=label,y_pred=modelPrediction)
    trainRecall(y_true=label,y_pred=modelPrediction)
    trainPrecision(y_true=label,y_pred=modelPrediction)
# define the Test function: 
@tf.function(input_signature=InputSignature)
def testStep(inputATensor,inputBTensor,label):
    """
    @brief Test the model performance on a batch of input data
    @details The function test the model using one batch of the input validation data. 
    @param inputATensor: tensor, a rank two tensor of shape (batch_size, seq_len) which contain 
    the tokenized sequences of the first protein. 
    @param inputBTensor: tensor, a rank two tensor of shape (batch_size, seq_len) which contain 
    the tokenized sequences of the second protein. 
    @param label: tensor, a rank one tensor which store for each pair of input sequences whether
    they interact of not.
    @note: During Accuracy calculations the model predictions are casted 
    as binary values, i.e. if the predictions are 0.5 AND above they are
    casted to One otherwise to zero.
    """
    modelPrediction=model(inputATensor,inputBTensor,False)
    loss=lossFunction(y_true=label,y_pred=modelPrediction)
    testLoss(loss)
    testAcuracy(y_true=label,y_pred=tf.cast(modelPrediction>=0.5,tf.int32))
    testAUC(y_true=label,y_pred=modelPrediction)
    testRecall(y_true=label,y_pred=modelPrediction)
    testPrecision(y_true=label,y_pred=modelPrediction)
# define a function to train the model: 
def trainEpoch(numEpoch,outputPath):
    """
    @brief train and evaluate the models for a specific number of epochs.
    @details The function trains the model for a specific number of epochs defined by 
    the variable numEpoch, after each epoch it update a record of the model
    performance on the training and test data-sets and it stores the weights of the
    embedding layer.
    @param numEpoch:int, it is the number of training epochs.
    @param outputPath: int, it is the path to store the model performance logs and the embedding layer weights. 
    @param embeddingWeightsDict: dict, it is a dict object to store the embedding layer 
        weights during the training. 
    """
    # a history object to store the data
    history=dict()
    for epoch in range(numEpoch):
     # reset the training metrics:
         trainLoss.reset_states()
         trainAcuracy.reset_states()
         trainAUC.reset_states()
         trainRecall.reset_states()
         trainPrecision.reset_states()
         # reset the test metrics: 
         testLoss.reset_states()
         testAcuracy.reset_states()
         testAUC.reset_states()
         testRecall.reset_states()
         testPrecision.reset_states()
    # define metric variables to store the results: 
         trainloss,trainauc,trainacc,trainrecall,trainprecision=0,0,0,0,0
         for (batch,(inputTensorA,inputTensorB,labels)) in enumerate(trainDataset):
             trainStep(inputTensorA,inputTensorB,labels)
             if batch%5==0: # print the model state to the console every 5 batch
                 print("Epoch: {}, Training Batch {} State: \n ".format(epoch,batch))
                 print("loss: {} \t Acc: {} \t AUC : {} \t Recall: {} \t Precision: {}\t".format(
                      trainLoss.result(),trainAcuracy.result(),
                      trainAUC.result(),trainRecall.result(),
                      trainPrecision.result()))
         trainloss=trainLoss.result().numpy()
         trainacc=trainAcuracy.result().numpy()
         trainauc=trainAUC.result().numpy()
         trainrecall=trainRecall.result().numpy()
         trainprecision=trainPrecision.result().numpy()
    # define metrics to store the evaluation results: 
         testloss,testauc,testacc,testrecall,testprecision=0,0,0,0,0
         for (batch,(inputTensorA,inputTensorB,labels)) in enumerate(testDataset):
             testStep(inputTensorA,inputTensorB,labels)
             print("Epoch: {}, Test Batch {} State: \n ".format(epoch,batch))
             print("loss: {} \t Acc: {} \t AUC : {} \t Recall: {} \t Precision: {}\t".format(
                      testLoss.result(),testAcuracy.result(),
                      testAUC.result(),testRecall.result(),
                      testPrecision.result()))
         testloss=testLoss.result().numpy()
         testacc=testAcuracy.result().numpy()
         testauc=testAUC.result().numpy()
         testrecall=testRecall.result().numpy()
         testprecision=testPrecision.result().numpy()
    # Write the results to tensorbord for Monitoring
    # Add the results to the history objects: 
         history[epoch]={
                 "train":{
                                    "loss":trainloss,
                                    "Accuracy":trainacc,
                                    "AUC":trainauc,
                                    "Recall":trainrecall,
                                    "Precision":trainprecision
                                            },
                "Test":{
                                "loss":testloss,
                                "Accuracy":testacc,
                                "AUC":testauc,
                                "Recall":testrecall,
                                "Precision":testprecision
                        }}
         # overwrite the model history object 
         with open(outputPath+".pickle","wb") as output_:
             pickle.dump(history,output_)
### Construct the experimental training loop:
## Construct the input Data sets:
numberOfTrainingExamples=firstProteinArrayTrain.shape[0]
trainDataset=tf.data.Dataset.from_tensor_slices(
        (
         firstProteinArrayTrain[0:int(dataFrac*numberOfTrainingExamples)],
         secondProteinArrayTrain[0:int(dataFrac*numberOfTrainingExamples)],
         thirdProteinsListTrain[0:int(dataFrac*numberOfTrainingExamples)])
        ).shuffle(1000).batch(2048)
        
testDataset=tf.data.Dataset.from_tensor_slices(
        (firstProteinArrayTest,secondProteinArrayTest,
         thirdProteinsListTest)).shuffle(1000).batch(250)

## Start the training loop
with tf.device('/gpu:0'):
     trainEpoch(100,outputPath)
print("******************* End Execution *******************")
             
             
             
             
             
