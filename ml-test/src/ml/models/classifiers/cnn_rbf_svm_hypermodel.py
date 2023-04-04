from .. import *
from ..cnnhypermodel import *

from sklearn.svm import SVC

## CNNRBFSVMHyperModel derived from AbstractCompositeClassifierHyperModel and CNNHyperModel
class CNNRBFSVMHyperModel(AbstractCompositeClassifierHyperModel, CNNHyperModel):

    ## Declare all hyperparameters for this model 
    def _declareParameters(self):
        super()._declareParameters()

        # Extend the list
        self._parameters.append(NumberParameter("gamma", min_value=1, max_value=3, step=5e-2, value=2))
        self._parameters.append(NumberParameter("C", min_value=1e-2, max_value=2, step=1e-2, value=1))

    ## Build the model with a values/hyperparameter dictionary 
    # @param values     The hyperparameters values dictionary
    # @return           Instance of a keras model
    def buildModel(self, values):
        # Build the model and classifier
        self._model = super().buildModel(values)
        self._classifier = SVC(
            kernel="rbf",
            gamma=values["gamma"],
            C=values["C"])

        return self._model
