from .choiceparameter import *
from .numberparameter import *


## Handler class for parameters inherited from AbstractParameter
class ParameterHandler:
    _params = list()
    
    ## Constructor for the ParameterHandler class
    def __init__(self):
        pass
    
    ## Add a parameter to this object
    # @param parameter  A parameter inherited from AbstractParameter
    def add(self, parameter):
        self._params.append(parameter)

    ## Set the list of parameters for this object
    # @param items
    def setList(self, items):
        self._params = items
   
    ## Get the list of parameters in this object
    # @return The list of parameters
    def getList(self):
        return self._params
    
    ## Get the number of parameters stored in this object
    # @return The number of parameters
    def getNumParameters(self):
        return len(self._params)
    
    ## Get a parameter by name
    # @param name    The name of the parameter
    # @return   The parameter if found
    def getParameter(self, name):
        for p in self._params:
            if p.getName() == name:
                return p

        return None
    
    ## Get a dictionary of values 
    # @param hp     A HyperParameters object from the keras tuner library
    # @return A dictionary of parameter values
    def getValuesDict(self, hp=None):
        if hp != None:
           items = self._createHyperparameters(hp) 
        else:
            items = dict()
            for p in self._params:
                items[p.getName()] = p.getValue()

        return items
    
    ## Create hyperparameters for the keras tuner library
    #
    # Fixed parameters are also stored as fixed parameters for the keras tuner library
    #
    # @param hp     A HyperParameters object from the keras tuner library
    # @return A dictionary containing keras tuner hyperparameters 
    def _createHyperparameters(self, hp):
        items = dict()

        for p in self._params:
            name = p.getName()

            if p.isFixed():
                hp.Fixed(name, p.getValue())

            if type(p) == ChoiceParameter:
                items[name] = hp.Choice(name, values=p.getOptions(), default=p.getValue()) 

            elif type(p) == NumberParameter:
                items[name] = hp.Int(name, min_value=p.getMinValue(), max_value=p.getMaxValue(), step=p.getStepSize(), default=p.getValue())

        return items
