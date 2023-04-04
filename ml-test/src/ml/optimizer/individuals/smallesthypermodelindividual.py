from .hypermodelindividual import * 

## Class derived from HyperModelIndividual for finding the smallest accurate hypermodel 
class SmallestHyperModelIndividual(HyperModelIndividual):
    ## Calculate fitness value for this individual
    # @param accuracy       The final accuracy value of the model
    # @param model_size     The number of parameters in the model
    def _calcFitness(self, accuracy, model_size):
        self._fitness = accuracy * (self._maxModelSize - model_size)
