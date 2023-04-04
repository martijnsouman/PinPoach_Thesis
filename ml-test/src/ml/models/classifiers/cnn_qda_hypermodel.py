from .. import *
from ..cnnhypermodel import *

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

## CNNQDAHyperModel derived from AbstractCompositeClassifierHyperModel and CNNHyperModel
class CNNQDAHyperModel(AbstractCompositeClassifierHyperModel, CNNHyperModel):

    ## Build the model with a values/hyperparameter dictionary 
    # @param values     The hyperparameters values dictionary
    # @return           Instance of a keras model
    def buildModel(self, values):
        # Build the model and classifier
        self._model = super().buildModel(values)
        self._classifier = QuadraticDiscriminantAnalysis() 

        return self._model
