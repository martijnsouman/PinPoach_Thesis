from .. import *
from ..cnnhypermodel import *

from sklearn.ensemble import AdaBoostClassifier


## CNNAdaBoostHyperModel derived from AbstractCompositeClassifierHyperModel and CNNHyperModel
class CNNAdaBoostHyperModel(AbstractCompositeClassifierHyperModel, CNNHyperModel):
    
    ## Declare all hyperparameters for this model 
    def _declareParameters(self):
        super()._declareParameters()

        # Extend the list
        self._parameters.append(NumberParameter("n_estimators", min_value=5, max_value=200, value=100))
        self._parameters.append(ChoiceParameter("learning_rate", options=[1.0, 1e-1, 1e-2, 1e-3, 1e-4], value=1))
        self._parameters.append(ChoiceParameter("algorithm", options=["SAMME", "SAMME.R"], value="SAMME"))

    ## Build the model with a values/hyperparameter dictionary 
    # @param values     The hyperparameters values dictionary
    # @return           Instance of a keras model
    def buildModel(self, values):
        # Build the model and classifier
        self._model = super().buildModel(values)
        self._classifier = AdaBoostClassifier(
            n_estimators=values["n_estimators"],
            learning_rate=values["learning_rate"],
            algorithm=values["algorithm"])

        return self._model
