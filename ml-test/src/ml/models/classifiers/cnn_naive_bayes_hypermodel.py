from .. import *
from ..cnnhypermodel import *

from sklearn.naive_bayes import GaussianNB

## CNNNaiveBayesHyperModel derived from AbstractCompositeClassifierHyperModel and CNNHyperModel
class CNNNaiveBayesHyperModel(AbstractCompositeClassifierHyperModel, CNNHyperModel):

    ## Declare all hyperparameters for this model 
    def _declareParameters(self):
        super()._declareParameters()

        # Extend the list
        self._parameters.append(ChoiceParameter("var_smoothing", options=[1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13], value=1e-9))

    ## Build the model with a values/hyperparameter dictionary 
    # @param values     The hyperparameters values dictionary
    # @return           Instance of a keras model
    def buildModel(self, values):
        # Build the model and classifier
        self._model = super().buildModel(values)
        self._classifier = GaussianNB(var_smoothing=values["var_smoothing"])

        return self._model
