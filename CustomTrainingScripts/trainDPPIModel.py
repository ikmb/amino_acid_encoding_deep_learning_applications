#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hesham ElAbd
@group Genetics and Bioinformatics group, IKMB, University of Kiel, Germany
@contact h.elabd@ikmb.uni-kiel.de
@date Mon Oct 28 14:57:51 2019

@brief A command-line script for reading and constructing a DPPI model @see DPPIBluePrint.py
  
@details Constructing a DPPI models with a user defined parameters, train it on a specific fraction of input data 
        using a specific GPU and then write the results dict as pickle object on a user defined object.
         The loss, metrics and model checkpoints are written a log directory which enable live monitoring of 
         the training using TensorBoard. 
"""
# load the modules:
import tensorflow as tf
import pickle
import os
from DPPIBluePrint import DPPIModel
import argparse
import datetime
## define the user input:
parser=argparse.ArgumentParser()

parser.add_argument('-n',
                    '--numTokens',
                    help='The number of input Tokens',
                    type=int)

parser.add_argument('-d',
                    '--embDim',
                    help='The output of the embedding dimension',
                    type=int)

parser.add_argument('-t',
                    '--trainEmbedding',
                    help="""a int of whether or not to train the weights of the
                            embedding layer after initialization. 1 indicates that the embeeding layer should be trained while 0
                            indicates that the weights should be frozen""",
                    type=int
                        )

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
numTokens=inputs.numTokens
embDim=inputs.embDim
trainEmbedding=inputs.trainEmbedding
gpuIndex=inputs.gpuIndex
outputPath=inputs.outputPath
dataFrac=inputs.dataFrac
# validating the user inputs
assert numTokens >=21, """ The number of unique tokens should be at least 21,
                            that is 20 amino acids + 1 """
                            
assert embDim>=1,"""The output of the embedding dimension should be at least one. 
Your input is: {} """.format(embDim)
                    
assert trainEmbedding == 0 or trainEmbedding==1, """ trainEmbedding should be 
0 or 1 where 1 indicates that the embeeding layer should be trained while 0
indicates that the weights should be frozen. Your input is: {} """.format(trainEmbedding)
                                
assert dataFrac >=0 and dataFrac<=1,""" data fraction should be a float between 
zero and one which specify The fraction of the training data to use.
Your input is: {} """.format(dataFrac)
          
# casting the trainEmbeeding as a bool
trainEmbedding=bool(trainEmbedding)
# Adjust the memory utilization by TF on GPUs:
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
model=DPPIModel(numTokens,embDim,trainEmbedding)
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
# define a function to build the models: 
@tf.function
def buildModel(inputATensor,inputBTensor):
    """
    @brief Build the weights of the model
    @details The function build the model by calling it with some inputs. The aim of doing 
    this is to initialize all of the model's weights. Hence, we can access the embedding
    matrix weights before we start training.
    @param inputATensor: tensor, a tensor of shape (batch_size,seq_len) which store
                         the encoded sequences of the first protein.
    @param inputBTensor: tensor, a tensor of shape (batch_size,seq_len) which store
                         the encoded sequences of the first protein.
    """
    _=model(inputATensor,inputBTensor,False) # calling the model with 
    # Training mode set to False
    model.summary() # print the summary of the model

# define a function to store the embeeding matrix:
def updateEmbeddingWeights(embeddingWeightsDict,epoch):
    """
    @brief add the weights of the embedding matrix at a specific epoch to a dict object.
    @details The function keeps a record of the Embedding weights after each epochs by 
    updating a dict object where keys are the epoch index and values are the 
    weights of the matrix. 
    @note the initial snap shot of the weights have the special key value -1. 
    @param embeddingWeightsDict: dict, a dict object to store the embedding matrix weights
    @param epoch: int, the epoch index to extract and append its weights the history-dict. 
    """
    weights=model.get_weights()[0]
    embeddingWeightsDict[epoch]=weights
    return embeddingWeightsDict

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
# define a tf.summary object to enable tensorboard visualization: 
currentTime=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
trainLogDir=outputPath+'/logs/DPPI_Model/'+currentTime+'/train'
testLogDir=outputPath+'/logs/DPPI_Model/'+currentTime+'/test'
graphDir=outputPath+'/logs/DPPI_Model/func/'+currentTime
trainSummaryWriter=tf.summary.create_file_writer(trainLogDir)
graphSummaryWriter=tf.summary.create_file_writer(graphDir)
testSummaryWriter=tf.summary.create_file_writer(testLogDir)
# define a checkpoint to store the model weights after each epochs: 
checkPoint=tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(checkPoint, graphDir,
                                          max_to_keep=100)
# define a function to train the model: 
def trainEpoch(numEpoch,outputPath,embeddingWeightsDict):
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
         # Write the results to tensorbord for Monitoring
         with trainSummaryWriter.as_default():
             tf.summary.scalar('Loss',trainLoss.result(),step=epoch)
             tf.summary.scalar('Accuracy',trainAUC.result(),step=epoch)
             tf.summary.scalar('AUC',trainAUC.result(),step=epoch)
             tf.summary.scalar('Recall',trainRecall.result(),step=epoch)
             tf.summary.scalar('Percision',trainPrecision.result(),step=epoch)
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
         with testSummaryWriter.as_default():
             tf.summary.scalar('Loss',testLoss.result(),step=epoch)
             tf.summary.scalar('Accuracy',testAUC.result(),step=epoch)
             tf.summary.scalar('AUC',testAUC.result(),step=epoch)
             tf.summary.scalar('Recall',testRecall.result(),step=epoch)
             tf.summary.scalar('Percision',testPrecision.result(),step=epoch)
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
         ckpt_manager.save()
         # write the Info of the models 
         if epoch%10==0: # Trace the function every 10 epoch by calling the 
             # model with the first 1000 element in the training dataset
             tf.summary.trace_on(graph=True, profiler=False)
             model(firstProteinArrayTrain[0:1000],
                  secondProteinArrayTrain[0:1000],False)
             with graphSummaryWriter.as_default():
                 tf.summary.trace_export(
                 name="ModelExecutionGraph"+str(epoch),
                 step=0,
                 profiler_outdir=graphDir)
         # store the embeeding weights after training 
         embeddingWeightsDict=updateEmbeddingWeights(embeddingWeightsDict,epoch)
         # overwrite the model history object 
         with open(outputPath+".pickle","wb") as output_:
             pickle.dump(history,output_)
### Construct the experimental training loop:
# define the weights of the embeddingWeightsDict
embeddingWeightsDict=dict()   
# build the model weights : 
buildModel(firstProteinArrayTrain[0:100],firstProteinArrayTrain[0:100])
# store the initialized weights: 
updateEmbeddingWeights(embeddingWeightsDict,-1)
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
     trainEpoch(100,outputPath,embeddingWeightsDict)
## save the embedding weights:
with open(outputPath+"/EmbeddingWeights.pickle","wb") as output_:
             pickle.dump(embeddingWeightsDict,output_)
print("******************* End Execution *******************")
             
             
             
             
             
