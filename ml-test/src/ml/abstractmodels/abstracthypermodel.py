from ..abstractmodels import * 
from ..parameters import *

#import keras_tuner
from keras_tuner import HyperParameters as hp
from keras_tuner import HyperModel
#from kerastuner import HyperModel
#from kerastuner import HyperParameters as hp

## AbstractHyperModel class
# 
# Used to extend the functionalities of the AbstractModel class with
# specific hypermodel model functions
class AbstractHyperModel(AbstractModel, HyperModel):
    _parameters = None
        
    ## Constructor for the AbstractHyperModel class
    # @param path           Path to a directory to load & store the model in
    # @param name           The name of the model
    # @param input_shape    The input data (train x) shape
    # @param verbose        Enable or disable verbose printing
    def __init__(self, path, name, input_shape, verbose=False):
        super().__init__(path, name, input_shape, verbose)

        self._parameters = list()
        self._declareParameters()
    
    ## Set the keras model object
    # @param model  The keras model object
    def setModel(self, model):
        self._model = model



    ## Clean the model
    def clean(self):
        # Remove the keras model
        self._model = None

        # Remove training history
        self._trainHistory = None

        # Clear the session to use less RAM
        keras.backend.clear_session()

        self._verbosePrint("Cleaned the model")
    
        

    ## Build the model with hyperparameters
    # @param hp     The keras HyperParameters object
    # @return A keras model build with the specified hyperparamers
    def build(self, hp):
        params = self._getKerasTunerHyperparameters(hp)  
        return self.buildModel(params)


    
    ## Get the values dict for the keras hyperparameters 
    #
    # Fixed parameters are also stored as fixed parameters for the keras tuner library
    #
    # @param hp     A HyperParameters object from the keras tuner library
    # @return A dictionary containing keras tuner hyperparameters 
    def _getKerasTunerHyperparameters(self, hp):
        items = dict()

        for p in self._parameters:
            name = p.getName()
            value = p.getValue()
            paramType = type(p)
            
            # Create fixed hp parameters
            if p.isFixed():
                hp.Fixed(name, value)
            
            # Create choice parameters
            if paramType == ChoiceParameter:
                items[name] = hp.Choice(name, values=p.getOptions(), default=value) 
            
            # Create number parameters
            elif paramType == NumberParameter:
                # Create integer parameters
                if p.getType() == int:
                    hpType = hp.Int
                elif p.getType() == float:
                    hpType = hp.Float
                
                # Create the parameter
                items[name] = hpType(name, min_value=p.getMinValue(), max_value=p.getMaxValue(), step=p.getStepSize(), default=value)
        
        return items
    

    ## Get the declared hyperparameters for this model
    # @return   list with parameters
    def getHyperParameters(self):
        return self._parameters


    
    ## Build the model with a values/hyperparameter dictionary (Abstract function)
    # @param values     The hyperparameters values dictionary
    def buildModel(self, values):
        raise Exception("Function 'buildModel' not implemented.")
    
    ## Declare all hyperparameters for this model (Abstract function)
    def _declareParameters(self):
        raise Exception("Function '_declareParameters' not implemented.")
