from .nnmodel import *
from .cnnmodel import *
from .cnnhypermodel import *
from .conv1dhypermodel import *
from .classifiers import *


## ModelFactory class for easy instantiation of model objects
class ModelFactory:
    
    ## Constructor for the ModelFactory class
    # @param path           Path to a directory to load & store the model in
    # @param input_shape    The input data (train x) shape
    # @param verbose        Enable or disable verbose printing
    def __init__(self, path, input_shape, verbose=False):
        self._path = path
        self._inputShape = input_shape
        self._verboseMode = verbose

    ## Get an instance of the NNModel class 
    # @param name           The name of the model
    # @return               An instance of the NNModel class
    def getNNModel(self, name):
        return NNModel(self._path, name, self._inputShape, self._verboseMode)

    ## Get an instance of the CNNModel class 
    # @param name           The name of the model
    # @return               An instance of the CNNModel class
    def getCNNModel(self, name):
        return CNNModel(self._path, name, self._inputShape, self._verboseMode)

    ## Get an instance of the CNNHyperModel class 
    # @param name           The name of the model
    # @return               An instance of the CNNHyperModel class
    def getCNNHyperModel(self, name):
        return CNNHyperModel(self._path, name, self._inputShape, self._verboseMode)

    ## Get an instance of the Conv1DHyperModel class 
    # @param name           The name of the model
    # @return               An instance of the Conv1DHyperModel class
    def getConv1DHyperModel(self, name):
        return Conv1DHyperModel(self._path, name, self._inputShape, self._verboseMode)


    ## Get an instance of a composed hypermodel classifier
    # @param name           The name of the model
    # @param classifier     The classifier class to use. A list of available
    # classifiers can be obtained with getClassifierList
    # @return               An instance of the classifier parameter class
    def getClassifierHyperModel(self, name, classifier):
        return classifier(self._path, name, self._inputShape, self._verboseMode)

    ## Get a list of available classifiers for getClassifierHyperModel
    # @return           Dictionaries with classifiers
    def getClassifierList(self):
        classifiers = {
            "neural_network": CNNClassicHyperModel,
            # "nearest_neighbors": CNNNeighborsHyperModel,
            # "linear_svm": CNNLinSVMHyperModel,
            # "rbf_svm": CNNRBFSVMHyperModel,
            #"gaussian_process": CNNGaussianProcessHyperModel,
            # "decision_tree": CNNDecisionTreeHyperModel,
            # "random_forest": CNNRandomForestHyperModel,
            # "adaboost": CNNAdaBoostHyperModel,
            # "naive_bayes": CNNNaiveBayesHyperModel,
            # "qda": CNNQDAHyperModel
        }
        return classifiers

