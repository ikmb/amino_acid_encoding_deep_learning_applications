#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hesham ElAbd
@group Genetics and Bioinformatics Group, IKMB, University of Kiel, Germany
@contact h.elabd@ikmb.uni-kiel.de
@date Tue Nov  5 13:23:41 2019

@brief The blueprint for building RNN-based peptide HLA-II interaction models

@details The script defines two functions for modeling peptide-HLA-II interactions, the first model which 
         uses embedding is defined and constructed using `buildHlaRNNModel` while the second function
         which does not use embedding and depend upon one-hot encoding is defined and constructed using 
         `buildHlaRNNModelOneHot`. Both models have the following basic architecture
         inputTensor ==> Embedding Layer* ==> LSTM ==> output neuron ==> PredictionHead. 
         
         *this layer is skipped, in case, one-hot encoding is used.
         
@note The models are constructed using the Keras functional API, and hence can be trained with `.fit` method.     
"""
# load the module
import tensorflow as tf
# define the model
def buildHlaRNNModel(input_len, numberOfTokens, embDim, trainEmb,
                         embWeights=None):
    """
    @brief The function build a RNN peptide-HLAII interaction model.    
    @param input_len: int,  the length of the padded input peptide.
    @param numberOfTokens:int, the number of tokens in the input +1, for example, 
    incase of convential amino acids the number of toekns should be 21, 
    i.e. (20 amino acids +1 ).
    @param embdim: int, it is the embedding dimension.
    @param trainEmb: int, it is bool of whether or not to train the embedding layer
    @param embWeights:tensor, a rank two tensor of shape (numberOfTokensXembdim) used to 
                    initialize the weights of the embedding layer. 
    @return model: a tf.Keras model, the model constructed using the user defined parameters.
    """
           
    assert input_len>9, """ The maximum length of the amino acid can not be 
    shorter than 9 amino acids. your current input length is {}""".format(input_len)
    
    assert embDim>=1, """ Embedding dimension must be at least one, 
    your input is {}""".format(embDim)
    
    assert isinstance(trainEmb,bool), """ train embeeding should be a bool of 
    whether or not to train the weights of the embedding layer. 
    your current value is {}""".format(trainEmb)
    
    inputLayer=tf.keras.layers.Input((input_len,),name="InputLayer")
    
    if embWeights is  None: 
            embeddingLayer=tf.keras.layers.Embedding(input_dim=numberOfTokens,
                                     output_dim=embDim,
                                     trainable=trainEmb,
                                      name="EmbeddingLayer")(inputLayer)
    else: 
         if numberOfTokens != embWeights.shape[0] or embDim != embWeights.shape[1]:
            raise ValueError("""
                             The dimension of the weights matrix does not match 
                             the numOfToken or the embedding dimention. The function
                             expected a matrix of shape {} by {}, however, the 
                             provided matrix has shape {} by {}""".format(
                             numberOfTokens,embDim,
                             embWeights.shape[0],
                             embWeights.shape[1]))
         embeddingLayer=tf.keras.layers.Embedding(input_dim=numberOfTokens,
                                             output_dim=embDim,
                                             trainable=trainEmb,
                                             weights=[embWeights])(inputLayer)
    
    lstm=tf.keras.layers.LSTM(units=12,name="LSTMLayer")(embeddingLayer)
    
    output=tf.keras.layers.Dense(units=1,activation="sigmoid",name="Dense")(lstm)
    
    model=tf.keras.models.Model(inputs=[inputLayer],outputs=[output])   
    return model


def buildHla_RNNModelOneHot(input_len, numberOfTokens):
    """
    @brief The function build an RNN peptide-HLAII interaction model and train it 
    on a one-hot encoded input.
    @param input_len: int, the length of the padded input peptide.
    @param numberOfTokens: int, the number of tokens in the input +1, for example, 
    incase of convential amino acids the number of toekns should be 21, 
    i.e. (20 amino acids +1 ). this is used to control the shape of the input layer. 
    @return model: a tf.Keras model, the model constructed using the user defined parameters.
    """
           
    assert input_len>9, """ The maximum length of the amino acid can not be 
    shorter than 9 amino acids. your current input length is {}""".format(input_len)
    inputLayer=tf.keras.layers.Input((input_len,numberOfTokens),name="InputLayer")
    
    lstm=tf.keras.layers.LSTM(units=12,name="LSTMLayer")(inputLayer)
    
    output=tf.keras.layers.Dense(units=1,activation="sigmoid",name="Dense")(lstm)
    
    model=tf.keras.models.Model(inputs=[inputLayer],outputs=[output])   
    return model


