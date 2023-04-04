from .. import *
from ..cnnhypermodel import *

from sklearn.svm import SVC

## CNNLinSVMHyperModel derived from AbstractCompositeClassifierHyperModel and CNNHyperModel
class CNNLinSVMHyperModel(AbstractCompositeClassifierHyperModel, CNNHyperModel):

    ## Declare all hyperparameters for this model 
    def _declareParameters(self):
        super()._declareParameters()

        # Extend the list
        self._parameters.append(NumberParameter("C", min_value=5e-3, max_value=2, step=5e-3))

    ## Build the model with a values/hyperparameter dictionary 
    # @param values     The hyperparameters values dictionary
    # @return           Instance of a keras model
    def buildModel(self, values):
        # Build the model and classifier
        self._model = super().buildModel(values)
        self._classifier = SVC(kernel="linear", C=values["C"])

        return self._model
