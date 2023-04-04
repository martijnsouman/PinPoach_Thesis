from .abstractparameter import *

## Choice Parameter class
# 
# Objects of this type contain a single value with different options/valid values
class ChoiceParameter(AbstractParameter):
    
    ## Constructor for ChoiceParameter class
    # @param name   The name of the parameter
    # @param options   List of options/valid values
    # @param value  The value of the parameter
    # @param fixed  Boolean: parameter is fixed
    def __init__(self, name, options, value=None, fixed=False):
        # Pick the first item if no value is provided
        if value == None:
            value = options[0]

        super().__init__(name, value, fixed)
        self._options = options

    ## Set a random value
    def setRandomValue(self):
        self._value = random.choice(self._options)
    
    ## Get a list of options/valid values for this parameter
    # @return   List: All options/valid values for this parameter
    def getOptions(self):
        return self._options
    
    ## Mutate the parameter by a factor
    # @param factor     The factor to mutate the ChoiceParameter by
    def mutate(self, factor):
        if self.isFixed():
            return

        index = self._options.index(self._value)
        newIndex = (index + int(factor)) % len(self._options)
        self._value = self._options[newIndex]
    
    ## Breed the parameter with another parameter
    # @param other The other parameter to breed with
    # @return Resulting ChoiceParameter object
    def breed(self, other):
        if self.isFixed():
            return self

        #Keep the value in the resulting object
        return ChoiceParameter(name=self._name, value=self._value, options=self._options)


