#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hesham ElAbd
@group Genetics and Bioinformatics group, IKMB, University of Kiel, Germany
@contact h.elabd@ikmb.uni-kiel.de
@date Tue Nov  5 13:51:26 2019

@brief   The implementation of the CNN-LSTM based models used for modeling peptide-HLAII interactions.
@details The scripts define two models for studying peptide-HLAII interaction, the first mode which 
         uses embeeding is defined and constructed using `buildHlaCNN_RNNModel` while the second function
         which does not use embedding and depend upon one-hot encoding is defined and constructed using 
         `buildHlaCNN_RNNModelOneHot`. Both models have the following basic architecture
         inputTensor ==> ConvLayer ==> LSTM ==> output neuron ==> PredictionHead
         
@note The models are constructed using the Keras functional API, and hence can be trained with `.fit` method.
"""
# load the module
import tensorflow as tf
# define the model
def buildHlaCNN_RNNModel(input_len, numberOfTokens, embDim, trainEmb,
                         embWeights=None):
    """
    @brief The function build a CNN-RNN peptide-HLAII interaction model.

    @param input_len: int, the length of the padded input peptide 
    @param numberOfTokens: int, the number of tokens in the input +1, for example, 
    in case of conventional amino acids the number of tokens should be 21, i.e. (20 amino acids +1 ).
    @param embdim: int, is the embedding dimension.
    @param trainEmb: bool, whether or not to train the embedding layer
    @param embWeights: tensor, a rank 2 tensor of shape(numberOfTokensXembdim) used to initialize the weights of the
                        embedding layer.
    @return model: a tf.Keras model, the model constructed using the user defined parameters
    """
           
    assert input_len>9, """ The maximum length of the amino acid can not be 
    shorter than 9 amino acids. your current input length is {}""".format(input_len)
    
    assert embDim>=1, """ Embedding dimension must be at least one, 
    your input is {}""".format(embDim)
    
    assert isinstance(trainEmb,bool), """ train embedding should be a bool of 
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
    
    cnn=tf.keras.layers.Conv1D(filters=36,padding="same",
                               kernel_size=9,strides=1,
                               name="1DConvLayer")(embeddingLayer)
    lstm=tf.keras.layers.LSTM(units=12,name="LSTMLayer")(cnn)
    
    output=tf.keras.layers.Dense(units=1,activation="sigmoid",name="Dense")(lstm)
    
    model=tf.keras.models.Model(inputs=[inputLayer],outputs=[output])   
    return model
  
def buildHlaCNN_RNNModelOneHot(input_len, numberOfTokens):
    """
    @brief The function build a CNN-RNN peptide-HLAII interaction model and train it 
    on a one-hot encoded input and hence no embedding layer is needed. 
    @param input_len: int, the length of the padded input peptide.
    @param numberOfTokens: int, the number of tokens in the input +1, for example, 
    incase of convential amino acids the number of toekns should be 21, 
    i.e. (20 amino acids +1 ). this is used to control the depth of the 
    one-hot encoding vectors. 
    @return model: a tf.Keras model, the model constructed using the model parameters.
    """
    assert input_len>9, """ The maximum length of the amino acid can not be 
    shorter than 9 amino acids. your current input length is {}""".format(input_len)
    
    inputLayer=tf.keras.layers.Input((input_len,numberOfTokens),name="InputLayer")
    
    cnn=tf.keras.layers.Conv1D(filters=36,padding="same",
                               kernel_size=9,strides=1,
                               name="1DConvLayer")(inputLayer)
    lstm=tf.keras.layers.LSTM(units=12,name="LSTMLayer")(cnn)
    
    output=tf.keras.layers.Dense(units=1,activation="sigmoid",name="Dense")(lstm)
    
    model=tf.keras.models.Model(inputs=[inputLayer],outputs=[output])   
    return model



