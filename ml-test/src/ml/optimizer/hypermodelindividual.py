from .genetic import * 
from ..abstractmodels import *


## HyperModelIndividual class, inherited from AbstractIndividual
#
# This class implements a optimizable hypermodel for the genetic optimizer
class HyperModelIndividual(AbstractIndividual):
    _buildFunction = None
    _fitFunction = None
    _maxSize = 0
    _model = None
    
    ## Constructor for the HyperModelIndividual class
    # @param parameters A list of parameters (such as AbstractParameter objects, can be returned by ParameterHandler.getList())
    # @param build_function Callback function to the build function. Build function should return a keras model
    # @param fit_function   Callback function to fit the keras model object
    # @param max_size   The maximum size of a model
    def __init__(self, parameters, build_function, fit_function, max_size):
        super().__init__(parameters)
        self._buildFunction = build_function
        self._fitFunction = fit_function
        self._maxSize = max_size 
    
    ## Evaluate this inidivual
    # 
    # Calculates the fitness of this individual
    def evaluate(self):

        #Get the hyperparameter values
        values = self.getValuesDict()

        print("Values:")
        for key,value in values.items():
            print("{k}: {v}".format(k=key, v=value))

        #Build the model 
        self._model = self._buildFunction(values)
        
        #Get the amount of parameters
        paramCount = self._model.count_params()

        #Don't evaluate the model if the model is too large
        if paramCount >= self._maxSize:
            self._fitness = 0.0
            print("Model rejected due to high parameter count, n={n}".format(n=paramCount))
            return

        #Evaluate the model
        val_accuracy = max(self._fitFunction(self._model).history['val_accuracy'])
        #self._fitness = val_accuracy * (self._maxSize - paramCount)
        self._fitness = val_accuracy
        
        #Print info
        print("Fitness: {f} (acc={a}, params={p})".format(f=self._fitness, a=val_accuracy, p=paramCount))
    
    ## Get the keras model
    # @return keras model
    def getModel(self):
        return self._model
