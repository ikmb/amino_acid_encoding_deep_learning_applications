#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hesham ElAbd
@group Genetics and Bioinformatics Group, IKMB, Univeristy of Kiel, Germany
@contact h.elabd@ikmb.uni-kiel.de
@date Mon Oct 28 11:58:06 2019

@brief The script contains the blueprint of the DPPI model.  
@details The script contains the functions to build DPPI models, Which is composite of two 
        parts, a siameses-like convolution network (sCNN), followed by a feed forward neural network 
        (FFN) which predicts the probability of interaction between two proteins.
        The siameses-like convolution network (sCNN) network is implemented as a linear 
        stack of convolutional modules, each convolutional module is composite as follow
        Convolutional layer ==> ReLU activation ==> BatchNormalization Layer ==> 
        AveragePooling layer. 
        
@note: The code is implemented using TensorFlow 2.0 syntax using the Keras API
@note: Training is carried out using gradient-tape a custom training loop
"""
# load TensorFlow: 
import tensorflow as tf
# define a each convolution block as a custom layer
class ConvBlock(tf.keras.layers.Layer):
    """
    @brief The implementation of a convolution block.
    
    """
    def __init__(self,numFilters,kernelSize,addGlobal=False,**kwargs):
        """
        @details create a user defined convolution block.
        
        @param numFilters: int, the number of filter in convID layer 
        @param kernelSize: int, the kernel size for the conv1D layer
        @param addGlobal: bool, whether of not to use global average pooling instead average pooling
        @param **kwargs: parameters being forwarded to the bass class. 
        """
        super(ConvBlock,self).__init__(**kwargs)
        
        assert numFilters>=1, """ number of filters 
                                should be bigger than one,
                                your input is {}""".format(numFilters)
                                
        self.numFilters=numFilters
        self.kernelSize=kernelSize
        self.addGlobal=addGlobal
        self.convLayer=tf.keras.layers.Conv1D(filters=numFilters,
                                              kernel_size=kernelSize,
                                              padding='same')
        self.relu=tf.keras.layers.ReLU()
        self.batchNorm=tf.keras.layers.BatchNormalization()
        if addGlobal:
            self.pooling=tf.keras.layers.GlobalAveragePooling1D()
        elif not addGlobal:
            self.pooling=tf.keras.layers.AveragePooling1D()
        else: 
            raise ValueError(""" addGlobal should be a bool, however, addGlobal
                             has type {}""".format(type(addGlobal)))
            
    def call(self, inputTensor,training):
        """
        @brief The calling logic of the layer
        
        @detail The layer takes a rank three input tensor, with the following shape
        (Batch_size, SeqLen, Depth) then it apply the following transformations
        inputTensor ==> Cov1D ==> Relu ==> BatchNormalizatin ==> Average pooling. 
        
        @param inputTensor: tensor, a rank three tensor containing the encoded protein sequences
        @param training: bool, whether or not to run the layer in the training or the inference mode
        @return outputTensor: tensor, a rank three tensor containing the features extracted from the inputTensor.
       
        @note: providing the training flag is important as the behaviour of the BatchNorm
        depends upon this flag. 
        """
        assert len(inputTensor.shape)==3,""" bad input shape,
        please check that your input is a 3D Tensor """
        inputTensor=self.convLayer(inputTensor)  
        inputTensor=self.relu(inputTensor)
        inputTensor=self.batchNorm(inputTensor,training=training)
        inputTensor=self.pooling(inputTensor)
        return inputTensor
    
    def get_config(self):
        """
        @details add the layer configuration to the model configuration
        """
        config=super(ConvBlock,self).get_config()
        config.update({"numberOfFilters": self.numFilters,
                     "KernelSize":self.kernelSize,
                     "GlobalPooling":self.addGlobal})
        return config

# define a convolution module that is composite of Four convolution blocks 
class ConvModule(tf.keras.layers.Layer):
    """
    @brief The implementation of the Convolutional module
    
    @details the convolutional module which is implemented as a linear stack of 
    convolutional blocks, defined above, the module uses four linearly-stacked
    convolutional blocks to map an input tensor of encoded amino acids into an output
    tensor of higher feature representation of the input tensor.
    """
    def __init__(self,**kwargs): 
        """
        @brief create a convolution module using the default values
        """
        super(ConvModule,self).__init__(**kwargs)
        self.blockOne=ConvBlock(numFilters=64,kernelSize=5,
                                addGlobal=False,name="ConvBlock1")
        
        self.blockTwo=ConvBlock(numFilters=128,kernelSize=7,
                                addGlobal=False,name="ConvBlock2")
        
        self.blockThree=ConvBlock(numFilters=256,kernelSize=9,
                                  addGlobal=False,name="ConvBlock3")
        
        self.blockFour=ConvBlock(numFilters=512,kernelSize=15,
                                  addGlobal=True,name="ConvBlock4")
    def call(self,inputTensor,training):
        """
        @brief The calling logic of the convolution module
        
        @param inputTensor: tensor, a rank three tensor containing the encoded protein sequences
        @param training:  bool, whether or not to run the layer in the training or the inference mode
        @return outputTensor: tensor, a rank two tensor containing the features extracted from the protein sequence
        """
        inputTensor=self.blockOne(inputTensor,training=training)
        inputTensor=self.blockTwo(inputTensor,training=training)
        inputTensor=self.blockThree(inputTensor,training=training)
        return self.blockFour(inputTensor,training=training)

        
    def get_config(self):
        """
        @details add the layer configuration to the model configuration
        """
        config=super(ConvModule,self).get_config()
        config.update({"NumberOfBlocks":5})
        return config
    
    
# define the model: 
class DPPIModel(tf.keras.models.Model):
    """
    @brief The implementation of the DPPI model
    
    @details The model is implemented as linear stack of two subnetworks, 
    the first a convolutional modules, defined above and the second is a
    a random project module which is simple, one layer feed-forward neural network. 
    Thus, the model is cabe represented as follow:
    inputTensorA--|                                       |--RandProjectionA-->|
                  | ==> Embedding Layer* ==>ConvModule ==>|                    |==>Multiply==> OutputNeuron ==> interaction probability
    inputTensorB--|                                       |--RandProjectionB-->|
    
    @note both input tensors are encoded using the same Embedding layer.
    @note Multiply referees two element-wise multiplication
    @note The inputs to conv modules are processed sequentially in the same feed-forward pass.  
    """
    def __init__(self,numOfToken, embDim,trainEmbLayer,embWeights=None,**kwargs):
        """
        @brief build the model using the user-defined parameters.
        @param numOfToken: int, The number of unique elements in the inputSequences +1
        @param embDim: int, the embedding dimensionality of the embedding layer
        @param trainEmbLayer: bool, whether or not to train the weights of the embedding layer
        @param embWeights: tensor, a rank two tensor with shape (numOfToken x embDim) used
                            to initialize the weights of the embedding layer.
        @param **kwargs:  parameters being forwarded to the bass class. 
        """
        super(DPPIModel,self).__init__()
        if embWeights is  None: 
            self.embedding=tf.keras.layers.Embedding(input_dim=numOfToken,
                                     output_dim=embDim,
                                     trainable=trainEmbLayer)
        else: 
           if numOfToken != embWeights.shape[0] or embDim != embWeights.shape[1]:
               raise ValueError("""
                             The dimension of the weights matrix does not match 
                             the numOfToken or the embedding dimention. The function
                             expected a matrix of shape {} by {}, however, the 
                             provided matrix has shape {} by {}""".format(
                             numOfToken,embDim,
                             embWeights.shape[0],
                             embWeights.shape[1]))
               
           self.embedding=tf.keras.layers.Embedding(input_dim=numOfToken,
                                     output_dim=embDim,
                                     trainable=trainEmbLayer,
                                     weights=[embWeights])
           
        self.convModule=ConvModule(name="ConvolutionModule") 
           
        self.randomProjectOne=tf.keras.layers.Dense(
                              units=512,activation='relu',use_bias=False,
                              trainable=False,name="RandomProjectionOne")    
        
        self.randomProjectTwo=tf.keras.layers.Dense(
                              units=512,activation='relu',use_bias=False,
                              trainable=False,name="RandomProjectionTwo")
        
        self.outputUnit=tf.keras.layers.Dense(units=1, activation="sigmoid",
                                              name="OutPutUnits")
    @tf.function # mark the function for compilation
    def call(self,inputTensorA,inputTensorB,training):
        """
        @brief The feed-forward logic of the network
        @param inputTensorA: tensor, the numerically encoded tensor for the first protein.
        @param inputTensorB: tensor, the numerically encoded tensor for the second protein.
        """
        proteinA=self.embedding(inputTensorA)
        proteinB=self.embedding(inputTensorB)
        proteinA=self.convModule(proteinA,training)
        proteinB=self.convModule(proteinB,training)
        proteinA=self.randomProjectOne(proteinA)
        proteinB=self.randomProjectTwo(proteinB)
        proteinAMultproteinB=tf.multiply(proteinA,proteinB)
        return self.outputUnit(proteinAMultproteinB)
