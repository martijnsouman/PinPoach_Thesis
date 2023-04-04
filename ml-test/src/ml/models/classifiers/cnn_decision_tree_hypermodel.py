from .. import *
from ..cnnhypermodel import *

from sklearn.tree import DecisionTreeClassifier

## CNNDecisionTreeHyperModel derived from AbstractCompositeClassifierHyperModel and CNNHyperModel
class CNNDecisionTreeHyperModel(AbstractCompositeClassifierHyperModel, CNNHyperModel):

    ## Declare all hyperparameters for this model 
    def _declareParameters(self):
        super()._declareParameters()

        # Extend the list
        self._parameters.append(NumberParameter("max_depth", min_value=2, max_value=100, value=5))
        self._parameters.append(ChoiceParameter("max_features", options=["auto", "log2"], value="auto"))

    ## Build the model with a values/hyperparameter dictionary 
    # @param values     The hyperparameters values dictionary
    # @return           Instance of a keras model
    def buildModel(self, values):
        # Build the model and classifier
        self._model = super().buildModel(values)
        self._classifier = DecisionTreeClassifier(
            max_depth=values["max_depth"],
            max_features=values["max_features"])

        return self._model
