import random

## Abstract Parameter class
class AbstractParameter:
    
    ## Constructor for AbstractParameter class
    # @param name   The name of the parameter
    # @param value  The value of the parameter
    # @param fixed  Boolean: parameter is fixed
    def __init__(self, name, value, fixed=False):
        self._name = name
        self._value = value
        self._fixed = fixed

    ## Set a random value
    def setRandomValue(self):
        pass
    
    ## Get the value of this parameter
    # @return   The value of this parameter
    def getValue(self):
        return self._value

    ## Get the name of this parameter
    # @return   The name of this parameter
    def getName(self):
        return self._name
    
    ## Check if the parameter is a fixed (non changeable) parameter
    # @return   Parameter is fixed
    def isFixed(self):
        return self._fixed
    
    ## Abstract function for mutatating the value of this parameter by a factor
    # @param factor The factor to mutate the value by
    def mutate(self, factor):
        pass
    
    ## Mutate the parameter by a factor
    # @param factor     The factor to mutate the parameter by
    def breed(self, other):
        pass
    
    ## Python magic method for casting this class to a string
    # @return   A string containing the name and value of this parameter
    def __str__(self):
        return "{name}: {value}".format(name=self._name, value=self._value)
