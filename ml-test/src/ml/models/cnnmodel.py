from ..abstractmodels import * 

## Implementation of AbstractConvolutionalModel 
class CNNModel(AbstractConvolutionalModel):
    
    ## Build the Convolutional Neural Network model
    def build(self):
        super().build()
        self._model = Sequential()

        #Convolution layer
        self._model.add(
            kl.Conv2D(filters=32, kernel_size=(3,3), activation='relu',
            input_shape=self._inputShape, data_format='channels_last',
            padding='same')
        )

        #2D max pooling layer
        self._model.add(kl.MaxPooling2D((2,2)))
        
        #Second convolutional layer
        self._model.add(kl.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
        self._model.add(kl.MaxPooling2D((2,2)))

        #Flatten the convolution data
        self._model.add(kl.Flatten())

        #Add Dense layers
        self._model.add(kl.Dense(500, activation='relu', kernel_initializer='he_uniform'))
        self._model.add(kl.Dense(250, activation='relu'))
        self._model.add(kl.Dense(1, activation='sigmoid'))

        self._compile()
    
    ## Compile the Convolutional Neural Network model
    def _compile(self):
        super()._compile(optimizer_name="sgd", learning_rate=0.01)
