from ..abstractmodels import *


## Implementation of AbstractModel
class NNModel(AbstractModel):
    
    ## Build the Neural Network model
    def build(self):
        super().build()
        self._model = Sequential()
        self._model.add(kl.Input(shape=self._inputShape))
        
        #Flatten the convolution data
        self._model.add(kl.Flatten())

        self._model.add(kl.Dense(500, activation='relu'))
        self._model.add(kl.Dense(250, activation='relu'))
        self._model.add(kl.Dense(1, activation='sigmoid'))

        self._compile()
    
    ## Compile the Neural Network model
    def _compile(self):
        super()._compile(optimizer_name="sgd", learning_rate=0.01)
