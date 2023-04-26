from ..abstractmodels import *

from keras.optimizers.schedules import *

import math


## Implementation of AbstractHyperModel and AbstractConvolutionalModel
class Conv1DHyperModel(AbstractHyperModel, AbstractConvolutionalModel):

    ## Declare all parameters used for this model
    def _declareParameters(self):
        maxPoolSize = 25
        maxConvLayers = math.floor(math.log(self._inputShape[0], maxPoolSize))
        maxDenseLayers = 5

        #self._parameters.append(ChoiceParameter("optimizer", value="adamax", options=['sgd', 'adam', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']))
        #self._parameters.append(ChoiceParameter("learning_rate", value=1e-2, options=[1e-2, 1e-3, 1e-4]))

        self._parameters.append(NumberParameter("num_conv_layers", min_value=1, max_value=maxConvLayers, value=1))
        self._parameters.append(NumberParameter("num_dense_layers", min_value=1, max_value=maxDenseLayers, value=1))
        self._parameters.append(NumberParameter("maxpool_size", value=maxPoolSize, min_value=2, max_value=2, fixed=True))

        for i in range(0, maxConvLayers):
            self._parameters.append(NumberParameter("conv_{}_filters".format(i), value=16, min_value=8, max_value=64, step=8))
            self._parameters.append(ChoiceParameter("conv_{}_activation".format(i), value='relu', options=['relu', 'sigmoid', 'tanh']))
            self._parameters.append(NumberParameter("conv_{}_kernel".format(i), value=10, min_value=2, max_value=100))

        for i in range(0, maxDenseLayers):
            self._parameters.append(NumberParameter("dense_{}_units".format(i), value=100, min_value=50, max_value=1000, step=50))
            self._parameters.append(ChoiceParameter("dense_{}_activation".format(i), value='relu', options=['relu', 'sigmoid', 'tanh']))


    
    ## Build the actual model
    # @param values     The hyperparameters values dictionary
    # @return           Keras model object
    def buildModel(self, values):

        #Clear the session to use less RAM
        keras.backend.clear_session()
    
        #Build the smodel 
        self._model = Sequential()

        #Add input layer
        self._model.add(kl.Input(shape=self._inputShape))
        
        #Add convolution layers
        for i in range(values["num_conv_layers"]):
            numFilters = values["conv_{}_filters".format(i)]
            activationFunction = values["conv_{}_activation".format(i)]
            
            convLayer = kl.Conv1D(
                filters=numFilters, 
                kernel_size=values["conv_{}_kernel".format(i)], 
                activation=activationFunction,
                data_format='channels_last',
                padding='same')
            
            #Add the layers
            self._model.add(convLayer)
            self._model.add(kl.MaxPooling1D(values["maxpool_size"]))
            self._model.add(kl.Dropout(0.5))
            self._model.add(kl.BatchNormalization())


        #Flatten the convolution data
        self._model.add(kl.Flatten())
        

        #Add Dense layers
        for i in range(values["num_dense_layers"]):
            numUnits = values["dense_{}_units".format(i)]
            activationFunction = values["dense_{}_activation".format(i)]

            self._model.add(kl.Dense(units=numUnits, activation=activationFunction, kernel_initializer='he_uniform'))
            self._model.add(kl.Dropout(0.5))

       
        #Add output layer
        self._model.add(kl.Dense(1, activation='sigmoid'))
        
        #Compile the model
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.1,
                decay_steps=10000,
                decay_rate=0.96,
                staircase=True)
        self._compile(optimizer_name="sgd", learning_rate=lr_schedule)
        
        #Clear the self._model variable
        model = self._model
        # self._model = None
        
        #Return the model
        return model
