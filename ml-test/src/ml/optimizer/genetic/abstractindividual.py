import numpy as np
import copy
import json
import os

from ...parameters import *

## AbstractIndividual class
#
# An abstract class derived from the ParameterHandler class,
# which can be used to implement a optimizable hypermodel for the genetic optimizer.
class AbstractIndividual:
    _fitness = None
    _parameters = list()
    _seed = None
    _id = 0 
    _path = None
    
    ## Constructor for the AbstractIndividual class
    # @param parameters A list of parameters (such as AbstractParameter objects)
    def __init__(self, parameters):
        self._parameters = parameters

    ## Copy this object
    # @return A copy of the self object
    def copy(self):
        c = copy.copy(self)
        c._parameters = copy.deepcopy(self._parameters)
        c._fitness = None
        c._id = 0
        return c
    

    ## Set the seed for the random generator
    # @param seed    The random generator seed
    def setSeed(self, seed):
        self._seed = seed
        np.random.seed(seed)


    ## Get the random seed
    # @return The random seed
    def getSeed(self):
        return self._seed


    ## Set the ID 
    # @param unique_id      The unique id of the individual in this generation
    def setId(self, unique_id):
        self._id = unique_id


    ## Get the unique id
    # @return               The unique identifier for this individual
    def getId(self):
        return self._id

    
    ## Abstract function used for evaluating the individual
    def evaluate(self):
        raise Exception("Function 'evaluate' is not implemented.")


    ## Store the individual to a file 
    # @param base_path       The directory where data is stored 
    def store(self, base_path):
        # Create the data to write
        data = {
            "seed": self._seed, 
            "fitness": self._fitness,
            "values": self.getParameterValuesDict()
        }
        
        # Store the path
        self._path = base_path
        
        # Write to a file
        path = os.path.join(self._path, "individual_{n}.json".format(n=self._id))
        with open(path, "w+") as jsonFile:
            json.dump(data, jsonFile)


    
    ## Load the individual from a file
    # @param base_path           The directory where data is stored 
    # @param unique_id      The unique id of the individual in this generation
    def load(self, base_path=None, unique_id=None):
        if unique_id != None:
            self._id = unique_id
        
        # Change the base path if base path is set
        if base_path != None:
            self._path = base_path
            
        # Load the data 
        path = os.path.join(self._path, "individual_{n}.json".format(n=self._id))
        f = open(path)
        data = json.load(f)
        f.close()
        
        # Restore the values
        self._seed = data['seed']
        self._fitness = data['fitness']
        
        # Set the parameters as fixed parameters; 
        # the type and other properties are not stored, only the name and value 
        self._parameters = list()
        for name, value in data['values'].items():
            self._parameters.append(AbstractParameter(name=name, value=value, fixed=True))

    
    ## Get the fitness of this individual
    # @return The fitness calculated by the evaluate function
    def getFitness(self):
        return self._fitness
    
    ## Mutate this individual with deviation
    # @param deviation  The mutation deviation
    def mutate(self, deviation):
        factors = np.random.normal(loc=0, scale=deviation, size=len(self._parameters))
        factors = [round(x) for x in factors]
        
        for i, item in enumerate(self._parameters):
            item.mutate(factors[i])
    
    ## Breed this individual with another individual
    # @param other  The other individual to breed with
    # @return   The resulting individual
    def breed(self, other):
        newParameters = list()

        for value in self._parameters:
            otherValue = other.getParameter(value.getName())
            newParameters.append(value.breed(otherValue))

        newIndividual = self.copy()
        newIndividual._parameters = newParameters
        return newIndividual
    
    ## Get a dictionary containing the values of the parameters
    # @return A dictionary containing values of the parameters
    def getParameterValuesDict(self):
        items = dict()
        for p in self._parameters:
            items[p.getName()] = p.getValue()

        return items

    ## Get a parameter by name
    # @param name    The name of the parameter
    # @return   The parameter if found
    def getParameter(self, name):
        for p in self._parameters:
            if p.getName() == name:
                return p

        return None
    
    ## Randomize the values for all (non fixed) parameters
    def randomizeParameters(self):
        for p in self._parameters:
            if not p.isFixed():
                # Randomize the parameter
                p.setRandomValue()
