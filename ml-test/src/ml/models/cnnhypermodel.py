from ..abstractmodels import *

import math

## Implementation of AbstractHyperModel and AbstractConvolutionalModel
class CNNHyperModel(AbstractHyperModel, AbstractConvolutionalModel):

    ## Declare all parameters used for this model
    def _declareParameters(self):
        # Fixed values 
        maxPoolSize = 2
        maxDenseLayers = 5
        
        smallestInputShape = min(list(self._inputShape[:-1]))
        maxConvLayers = math.floor(math.log(smallestInputShape, maxPoolSize))
        
        # Number of layers
        self._parameters.append(NumberParameter("num_conv_layers", min_value=1, max_value=maxConvLayers, value=3))
        self._parameters.append(NumberParameter("num_dense_layers", min_value=1, max_value=maxDenseLayers, value=4))
        
        # Optimizer parameters
        self._parameters.append(ChoiceParameter("optimizer", value="adamax", options=['sgd', 'adam', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']))
        self._parameters.append(ChoiceParameter("learning_rate", value=1e-4, options=[1e-2, 1e-3, 1e-4]))

        # Convolutional layer parameters
        self._parameters.append(NumberParameter("maxpool_size", value=maxPoolSize, min_value=2, max_value=2, fixed=True))
        for i in range(0, maxConvLayers):
            self._parameters.append(NumberParameter("conv_{}_filters".format(i), value=16, min_value=8, max_value=64, step=4))
            self._parameters.append(ChoiceParameter("conv_{}_activation".format(i), value='relu', options=['relu', 'sigmoid', 'tanh']))
            self._parameters.append(NumberParameter("conv_{}_kernel_x".format(i), value=10, min_value=3, max_value=10))
            self._parameters.append(NumberParameter("conv_{}_kernel_y".format(i), value=10, min_value=3, max_value=10))
        
        # Dense layer parameters
        for i in range(0, maxDenseLayers):
            self._parameters.append(NumberParameter("dense_{}_units".format(i), value=100, min_value=50, max_value=500, step=50))
            self._parameters.append(ChoiceParameter("dense_{}_activation".format(i), value='relu', options=['relu', 'sigmoid', 'tanh']))
            

    
    ## Build the actual model
    # @param values     The hyperparameters values dictionary
    # @return           Keras model object
    def buildModel(self, values):
        # Build the model 
        self._model = Sequential()
    
        # add input layer
        self._model.add(kl.InputLayer(self._inputShape))
        
        # add convolution layers
        for i in range(values["num_conv_layers"]):
            numFilters = values["conv_{}_filters".format(i)]
            activationFunction = values["conv_{}_activation".format(i)]
            
            kernelX = values["conv_{}_kernel_x".format(i)]
            kernelY = values["conv_{}_kernel_y".format(i)]
            kernelSize = (kernelX, kernelY)
            
            convLayer = kl.Conv2D(
                filters=numFilters, 
                kernel_size=kernelSize, 
                activation=activationFunction,
                data_format='channels_last',
                padding='same')
            
            # add the layers
            self._model.add(convLayer)
            self._model.add(kl.MaxPooling2D(values["maxpool_size"]))


        # Flatten the convolution data
        self._model.add(kl.Flatten())

        # add Dense layers
        for i in range(values["num_dense_layers"]):
            numUnits = values["dense_{}_units".format(i)]
            activationFunction = values["dense_{}_activation".format(i)]
            self._model.add(kl.Dense(units=numUnits, activation=activationFunction, kernel_initializer='he_uniform'))

       
        # add output layer
        self._model.add(kl.Dense(1, activation='sigmoid'))
        
        # Compile the model
        self._compile(optimizer_name=values["optimizer"], learning_rate=values["learning_rate"])
        
        # Clear the self._model variable
        model = self._model
        #self._model = None
        
        # Return the model
        return model
