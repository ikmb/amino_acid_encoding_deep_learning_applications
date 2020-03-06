#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hesham El Abd
@brief: prepear the input Tensor for Protein Protein Interaction Models.
"""
# load the modules': 
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from random import sample
import pandas as pd
from utils import shuffleArray, splitSequenceintoKmers
import os
# load the binnarized dataset: 
with open("TrembleDataBase.pickle","rb") as input_:
    database_Tremble=pickle.load(input_)
with open("SwissDataBase.pickle","rb") as input_:
    database_Swiss=pickle.load(input_)
with open('unique_protein_set.pickle','rb') as input_:
    database_unique=pickle.load(input_)
# Checking the validitiy of the keys:
curratedKeys=set()
unmappedKeys=set()
TrembleKeys=database_Tremble.keys()
SwissKeys=database_Swiss.keys()
for key in database_unique:
    if key in SwissKeys:
        curratedKeys.add(key)
    else:
        if key in TrembleKeys:
            curratedKeys.add(key)
        else:
            unmappedKeys.add(key)

print("Number of mapped keys: "+str(len(curratedKeys)))
print("Number of unmapped keys: "+str(len(unmappedKeys)))
print("Success Percentage: "+str(
        (len(curratedKeys)/len(database_unique))*100)+"%")
## Extracting the protein sequences:
curratedPPIDataBase=dict()
for key in curratedKeys: 
    try:
        curratedPPIDataBase[key]=database_Swiss[key]
    except: 
        try:
            curratedPPIDataBase[key]=database_Tremble[key]
        except: 
            print("I can not Extract the sequence of "+key+" !")
            pass
print("Sequences have been extracted")
print("Saving the generated database locally")
with open("curratedDataBase.pickle","wb") as output_:
    pickle.dump(curratedPPIDataBase,output_)
## Cleaning the database by length:
sequences_length=[]
for _, seq in curratedPPIDataBase.items():
    sequences_length.append(len(seq))
sequences_length=np.array(sequences_length)
print("cleaning the protein DataBase by length: ")
print("\t Removing protein shorter than 100 amino acid")
print(" \t Removing protein longer than 1000 amino acids")
cleanedByLength=dict()
for protID, protSeq in curratedPPIDataBase.items():
    if (len(protSeq)>=100 and len(protSeq)<=1000):
        cleanedByLength[protID]=protSeq
print("Cleaning the DataBase contains "+str(
        (len(cleanedByLength)/len(curratedPPIDataBase)*100)
        )+"% of the uncleaned database")
print("Cleaning Protein Protein Interaction DataBase")
# load the database: 
with open("set_pairs.pickle","rb") as input_:
    interactingPair=pickle.load(input_)
## Clean The DataBase: 
cleanedInteractingPair=set()
cleanedKeys=cleanedByLength.keys()
for pair in interactingPair:
    if (pair[0] in cleanedKeys and pair[1] in  cleanedKeys):
        cleanedInteractingPair.add(pair)
print("Number of pairs before cleaning: "+str(len(interactingPair)))
print("Number of pairs after cleaning: "+str(len(cleanedInteractingPair)))
print("Number of unique proteins:"+str(len(cleanedKeys)))
print("Reduction Ration: "+str(1-(len(cleanedInteractingPair)/len(interactingPair))))
print("writing the cleaning database to the disk")
with open("CleanedInteractionPaired.pickle","wb") as output_:
    pickle.dump(cleanedInteractingPair,output_)
## Split the unique identifer set into training and test datasets: 
trainDataSetUniqueIds, testDataSetUniqueIds = train_test_split(
        list(cleanedKeys),test_size=0.1)
trainDataSetUniqueIds=set(trainDataSetUniqueIds)
testDataSetUniqueIds=set(testDataSetUniqueIds)
## Writing the Results to the disk: 
with open("trainDataSetUniqueIds.pickle","wb") as output_:
    pickle.dump(trainDataSetUniqueIds,output_)

with open("testDataSetUniqueIds.pickle","wb") as output_:
    pickle.dump(testDataSetUniqueIds,output_)    
## Remove by Homology: 
# extract the sequences of the training proteins 
trainSequences=dict()
for trainProteinId in trainDataSetUniqueIds:
    trainSequences[trainProteinId]=cleanedByLength[trainProteinId]
# extract the sequences pf the test dataset: 
testSequences=dict()
for testProteinId in testDataSetUniqueIds:
    testSequences[testProteinId]=cleanedByLength[testProteinId]
# write the training dataset as a FASTA file: 
with open("trainingSequencesDataSet.fasta","w") as output_:
    for proteinId,proteinSeq in trainSequences.items():
        output_.write(">"+proteinId+" \n")
        output_.write(proteinSeq+" \n")
# wite the test database: 
with open("testingSequencesDataSet.fasta","w") as output_:
    for proteinId,proteinSeq in testSequences.items():
        output_.write(">"+proteinId+" \n")
        output_.write(proteinSeq+" \n")        
print("*********************************************************************")
print ("** Blast the Test dataset against the train dataset using blastp")
# Note: 
# Blasting using a system call
os.system(" ./ncbi-blast-2.9.0+/bin/makeblastdb -in trainingSequencesDataSet.fasta -title TrainingDB -dbtype prot ")
os.system("  ./ncbi-blast-2.9.0+/bin/blastp -query testingSequencesDataSet.fasta -out testBlastedTrain.csv -db trainingSequencesDataSet.fasta -outfmt 6 -num_threads 32 ")
print("**********************************************************************")
print("**** Parsing the blast results")
# Read the FASTA files: 
testBlastResult=pd.read_csv("testBlastedTrain.csv", sep="\t",header=None)
# constract a dictionary that contians the homology percentage perprotein
maxHomologyPerTestProtein=dict()
for rowIndx in range(testBlastResult.shape[0]):
    if testBlastResult.iloc[rowIndx,0] in maxHomologyPerTestProtein.keys():
        maxHomologyPerTestProtein[testBlastResult.iloc[rowIndx,0]]=max(
                maxHomologyPerTestProtein[testBlastResult.iloc[rowIndx,0]],
                testBlastResult.iloc[rowIndx,2])
    else: 
        maxHomologyPerTestProtein[testBlastResult.iloc[rowIndx,0]]=testBlastResult.iloc[rowIndx,2]
# Extract proteins that have homology of 40% and less:
cleanedByHomologyTestDataSet=dict()
for proteinId, homologyRatio in maxHomologyPerTestProtein.items():
    if homologyRatio<40:
        cleanedByHomologyTestDataSet[proteinId]=homologyRatio
testDataSetUniqueIds=cleanedByHomologyTestDataSet.keys()
## Extract the test and training datasets
    
trainingDataSetUniquePairs,testDataSetUniquePairs=set(),set()
# construct the training dataset: 
for interactingPair in cleanedInteractingPair: 
    if (interactingPair[0] in trainDataSetUniqueIds and 
        interactingPair[1] in trainDataSetUniqueIds):
        trainingDataSetUniquePairs.add(interactingPair)
# construct the test dataset: 
for interactingPair in cleanedInteractingPair: 
    if (interactingPair[0] in testDataSetUniqueIds and 
        interactingPair[1] in testDataSetUniqueIds):
        testDataSetUniquePairs.add(interactingPair)
## Write The results to the disk: 
with open("trainingDataSetUniquePairs.pickle","wb") as output_:
    pickle.dump(trainingDataSetUniquePairs,output_)
    
with open("testDataSetUniquePairs.pickle","wb") as output_:
    pickle.dump(testDataSetUniquePairs,output_)    
### Extract protein sequences: 
## First Training Proteins:
firstProteinTrain, secondProteinTrain =[],[]
idsTrainDataSetSequence=cleanedByLength.keys()  
for proteinPair in  trainingDataSetUniquePairs:
    if (proteinPair[0] in idsTrainDataSetSequence and 
        proteinPair[1] in idsTrainDataSetSequence):
        firstProteinTrain.append(cleanedByLength[proteinPair[0]])
        secondProteinTrain.append(cleanedByLength[proteinPair[1]])
assert len(firstProteinTrain)==len(secondProteinTrain)
labelsPositiveTrain=np.ones((len(firstProteinTrain))).reshape(-1,1)
# construct negative examples: 
firstProteinTrainNeg,secondProteinTrainNeg=[],[]
for _ in range(len(firstProteinTrain)):
    firstProteinTrainNeg.append(sample(firstProteinTrain,1)[0])
    secondProteinTrainNeg.append(sample(secondProteinTrain,1)[0])
assert len(firstProteinTrainNeg)==len(secondProteinTrainNeg) 
labelsNegativeTrain=np.zeros(len(firstProteinTrainNeg)).reshape(-1,1)

# assemble the training DataSet: 
firstProteinsListTrain=firstProteinTrain+firstProteinTrainNeg
secondProteinsListTrain=secondProteinTrain+secondProteinTrainNeg
thirdProteinsTensorTrain=np.concatenate((labelsPositiveTrain,labelsNegativeTrain),
                               axis=0)
assert len(firstProteinsListTrain)==thirdProteinsTensorTrain.shape[0]
assert len(secondProteinsListTrain)==thirdProteinsTensorTrain.shape[0]

## Second Test dataSet: 
firstProteinTest, secondProteinTest =[],[]
idsCleanTestDataSetSequence=cleanedByLength.keys()  
for proteinPair in  testDataSetUniquePairs:
    if (proteinPair[0] in idsCleanTestDataSetSequence and 
        proteinPair[1] in idsCleanTestDataSetSequence):
        firstProteinTest.append(cleanedByLength[proteinPair[0]])
        secondProteinTest.append(cleanedByLength[proteinPair[1]])
assert len(firstProteinTest)==len(secondProteinTest)
labelsPositiveTest=np.ones(len(firstProteinTest)).reshape(-1,1)
## Generate The Negative Examples for the Test Dataset: 
firstProteinTestNeg,secondProteinTestNeg=[],[]
for _ in range(len(firstProteinTest)):
    firstProteinTestNeg.append(sample(firstProteinTest,1)[0])
    secondProteinTestNeg.append(sample(secondProteinTest,1)[0])
assert len(firstProteinTestNeg)==len(secondProteinTestNeg) 
labelsNegativeTest=np.zeros(len(firstProteinTestNeg)).reshape(-1,1)
## Assemble the test dataset: 
firstProteinsListTest=firstProteinTest+firstProteinTestNeg
secondProteinsListTest=secondProteinTest+secondProteinTestNeg
thirdProteinsTensorTest=np.concatenate((labelsPositiveTest,labelsNegativeTest),
                               axis=0)
assert len(firstProteinsListTest)==thirdProteinsTensorTest.shape[0]
assert len(firstProteinsListTest)==thirdProteinsTensorTest.shape[0]
print("***** parsing the amino acids with k=1 *****")
# construct a protein sequence corpa for tokenization 
proteinCorpaForTokenizer=firstProteinTest+secondProteinTest+secondProteinTrain+firstProteinTrain
# define the tokenizer:
tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=21, 
                                                char_level=True)
"""
***Notice:
    Within the protein corpa there is three extra amino acids beside the 
    standard 20 amino acids namely, X, U and B. Including them in the tokenizer
    is straightforward forward which can be done with the command :
        tf.keras.preprocessing.text.Tokenizer(num_words=24, 
                                                char_level=True,
                                                filters=None)
        however, as other encoding schemes do not mostly support these 
        unconventional amino acids, we restrict the the tokeneization 
        processes to only focus on the conventional amino acids. Hence
        these amino acids are treated as "out of vocabulaory token", i.e. 
        OOV_tokens and are completely ignored by the tokenizer during the 
        encoding process.   
***
"""
# Fitting the tokenizer to the Protein Corpa: 
tokenizer.fit_on_texts(proteinCorpaForTokenizer)
# save a copy of the tokenizer for later usages: 
with open("OneMerAminoAcidTokenizer.pickle","wb") as output_:
    pickle.dump(tokenizer,output_)
## Encode the train and the test dataset as input tensors:
firstProteinArrayTrain=tf.keras.preprocessing.sequence.pad_sequences(
        sequences=tokenizer.texts_to_sequences(firstProteinsListTrain),
        dtype="int32",maxlen=1000,padding="pre")

secondProteinArrayTrain=tf.keras.preprocessing.sequence.pad_sequences(
        sequences=tokenizer.texts_to_sequences(secondProteinsListTrain),
        dtype="int32",maxlen=1000,padding="pre")

firstProteinArrayTest=tf.keras.preprocessing.sequence.pad_sequences(
        sequences=tokenizer.texts_to_sequences(firstProteinsListTest),
        dtype="int32",maxlen=1000,padding="pre")

secondProteinArrayTest=tf.keras.preprocessing.sequence.pad_sequences(
        sequences=tokenizer.texts_to_sequences(secondProteinsListTest),
        dtype="int32",maxlen=1000,padding="pre")
## shuffle the arrays: 
firstProteinArrayTrain,secondProteinArrayTrain,thirdProteinsTensorTrain=shuffleArray(
        firstProteinArrayTrain,secondProteinArrayTrain,thirdProteinsTensorTrain)

firstProteinArrayTest,secondProteinArrayTest,thirdProteinsTensorTest=shuffleArray(
        firstProteinArrayTest,secondProteinArrayTest,thirdProteinsTensorTest) 
## write the Results to the harddisk for later usage: 
with open("OneKmerArrayOneTrainEqual.pickle","wb") as output_:
    pickle.dump(firstProteinArrayTrain,output_)

with open("OneKmerArrayTwoTrainEqual.pickle","wb") as output_:
    pickle.dump(secondProteinArrayTrain,output_)
    
with open("OneKmerArrayThreeTrainEqual.pickle","wb") as output_:
    pickle.dump(thirdProteinsTensorTrain,output_)

with open("OneKmerArrayOneTestEqual.pickle","wb") as output_:
    pickle.dump(firstProteinArrayTest,output_)

with open("OneKmerArrayTwoTestEqual.pickle","wb") as output_:
    pickle.dump(secondProteinArrayTest,output_)

with open("OneKmerArrayThreeTestEqual.pickle","wb") as output_:
    pickle.dump(thirdProteinsTensorTest,output_)
print("**************** End of Data PreProcessing Stage ********************")













