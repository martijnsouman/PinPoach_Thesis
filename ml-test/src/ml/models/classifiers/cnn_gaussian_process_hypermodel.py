from .. import *
from ..cnnhypermodel import *

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

## CNNGaussianProcessHyperModel derived from AbstractCompositeClassifierHyperModel and CNNHyperModel
class CNNGaussianProcessHyperModel(AbstractCompositeClassifierHyperModel, CNNHyperModel):

    ## Build the model with a values/hyperparameter dictionary 
    # @param values     The hyperparameters values dictionary
    # @return           Instance of a keras model
    def buildModel(self, values):
        # Build the model and classifier
        self._model = super().buildModel(values)
        self._classifier = GaussianProcessClassifier(1.0 * RBF(1.0))

        return self._model

