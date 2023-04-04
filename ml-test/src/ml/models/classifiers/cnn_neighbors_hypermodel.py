from .. import *
from ..cnnhypermodel import *

from sklearn.neighbors import KNeighborsClassifier

## CNNNeighborsHyperModel derived from AbstractCompositeClassifierHyperModel and CNNHyperModel
class CNNNeighborsHyperModel(AbstractCompositeClassifierHyperModel, CNNHyperModel):

    ## Declare all hyperparameters for this model 
    def _declareParameters(self):
        super()._declareParameters()

        # Extend the list
        self._parameters.append(NumberParameter("n_neighbors", min_value=1, max_value=100, value=5))
        self._parameters.append(ChoiceParameter("weights", options=["uniform", "distance"], value="uniform"))
        self._parameters.append(ChoiceParameter("algorithm", options=["ball_tree", "kd_tree", "brute"], value="ball_tree"))
        self._parameters.append(NumberParameter("leaf_size", min_value=5, max_value=100, value=30))

    ## Build the model with a values/hyperparameter dictionary 
    # @param values     The hyperparameters values dictionary
    # @return           Instance of a keras model
    def buildModel(self, values):
        # Build the model and classifier
        self._model = super().buildModel(values)
        self._classifier = KNeighborsClassifier(
            n_neighbors=values["n_neighbors"],
            weights=values["weights"],
            algorithm=values["algorithm"],
            leaf_size=values["leaf_size"])

        return self._model
