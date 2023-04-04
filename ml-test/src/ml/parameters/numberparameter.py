from .abstractparameter import *


## NumberParameter class
#
# Objects of this type can contain a number as value.
# A number has a minimum value, maximum value and a step size
class NumberParameter(AbstractParameter):
    
    ## Constructor for NumberParameter class
    # @param name   The name of the parameter
    # @param min_value   The minimum value of this parameter
    # @param max_value   The maximum value of this parameter
    # @param value  The value of the parameter
    # @param step   The step size between values in the range min, max
    # @param fixed  Boolean: parameter is fixed
    def __init__(self, name, min_value, max_value, value=None, step=1, fixed=False):
        # If no value is provided, set value to the minimum value
        if value == None:
            value = min_value
            
        super().__init__(name, value, fixed)
        self._minValue = min_value
        self._maxValue = max_value
        self._stepSize = step

        # Check what number type to use
        types = [type(x) for x in [self._minValue, self._maxValue, self._stepSize]]

        if float in types:
            self._type = float
        elif int in types:
            self._type = int
        else:
            raise ValueError("Unsupported type in number parameter: " + str(types))

    ## Set a random value
    def setRandomValue(self):
        # Use a scaling factor because the
        # randrange function only accepts integer values
        if self._type == float:
            scale = int(1/self._stepSize)
            self._value = random.randrange(
                start=int(self._minValue*scale),
                stop=int((self._maxValue+self._stepSize)*scale),
                step=1)/scale
        else:
            self._value = random.randrange(
                start=self._minValue, 
                stop=self._maxValue+self._stepSize,
                step=self._stepSize)
    
    ## Get the type of this parameter 
    # @return   The parameter is a float or an int
    def getType(self):
        return self._type
    
    ## Get the minimum value of this parameter
    # @return The minimum value
    def getMinValue(self):
        return self._minValue

    ## Get the maximum value of this parameter
    # @return The maximum value
    def getMaxValue(self):
        return self._maxValue
    
    ## Get the step size of this parameter
    # @return The step size 
    def getStepSize(self):
        return self._stepSize
    
    ## Mutate the parameter by a factor
    # @param factor     The factor to mutate the NumberParameter by
    def mutate(self, factor):
        if self.isFixed():
            return
        
        self._value += self._type(self._type(factor) * self._stepSize)
        
        #Check if the value is still in range
        if self._value > self._maxValue:
            self._value = self._maxValue
        elif self._value < self._minValue:
            self._value = self._minValue


    ## Breed the parameter with another parameter
    # @param other The other parameter to breed with
    # @return Resulting NumberParameter object
    def breed(self, other):
        if self.isFixed():
            return self

        selfMultiplier = (self._value - self._minValue)/self._stepSize
        otherMultiplier = (other._value - other._minValue)/other._stepSize
        newMultiplier = int((selfMultiplier + otherMultiplier)/2)
        value = newMultiplier * self._stepSize
        
        #Return the new resulting object
        return NumberParameter(
                name=self._name, 
                min_value=self._minValue,
                max_value=self._maxValue,
                value=value,
                step=self._stepSize
        )
