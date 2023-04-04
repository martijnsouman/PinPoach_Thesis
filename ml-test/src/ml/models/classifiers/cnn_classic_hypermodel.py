from .. import *
from ..cnnhypermodel import *

## CNNClassicHyperModel derived from CNNHyperModel for compatibility with the optimizer
class CNNClassicHyperModel(CNNHyperModel):

    ## Build the model with a values/hyperparameter dictionary 
    # @param values     The hyperparameters values dictionary
    # @return           Instance of a keras model
    def buildModel(self, values):
        # Build the model and classifier
        self._model = super().buildModel(values)

        return self._model
