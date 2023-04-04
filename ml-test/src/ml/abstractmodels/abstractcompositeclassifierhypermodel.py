from ..abstractmodels import *
from joblib import dump, load

## AbstractCompositeClassifierHyperModel class
# 
# Can be used derive classed from to create composed classifiers
class AbstractCompositeClassifierHyperModel(AbstractHyperModel):
    _classifier = None

    ## Clean the model
    def clean(self):
        self._classifier = None
        super().clean()
    
    ## Modify the model to obtain the feature vector model
    # @param feature_vector_layer       The layer to use as a feature vector
    # @return                           The modified model
    def _getFeatureVectorModel(self, feature_vector_layer):
        return Model(self._model.input, self._model.get_layer(feature_vector_layer).output)
    
    ## Get the feature vector data from the already modified model
    # Modify the model first by usign _getFeatureVectorModel
    # @param x      Input dataset
    # @return       Output dataset, determined by the feature vector model
    def _getFeatureVectorData(self, x):
        # Get the output values from the feature vector model
        return self._model.predict(x)

    ## Train the model
    # @param train_x            The input training set
    # @param train_y            The output training set
    # @param epoch_count        The amount of epochs to train the model
    # @param val_split          The percentage of which the train data is split into a separate validation set
    # @param epochs_patience    If the accuracy does not improve after this many epochs, the training will be stopped
    # @param callbacks          A list of keras callbacks instances. If this parameter is provided, the epochs_patience 
    # paramter is ignored.
    # @param batch_size         The size of a training batch
    def train(self, train_x, train_y, epoch_count=100, val_split=0.1, epochs_patience=10, callbacks=None, batch_size=32):
        # Train the model
        super().train(train_x, train_y, epoch_count, val_split, epochs_patience, callbacks, batch_size)

        # Build the feature vector model
        self._model = self._getFeatureVectorModel("flatten")
        
        # Get the feature vector data
        x = self._getFeatureVectorData(train_x)
        
        # Fit the classifier
        self._classifier.fit(x, train_y)

    
    ## Determine the accuracy of the model for the test dataset
    # @param input_data     The input test set
    # @param output_data     The output test set
    # @return scores, confusionMatrix, predictions list
    def evaluate(self, input_data, output_data):
        self._verbosePrint("Evaluating model..")
        
        # Predict values
        predictions = self.predict(input_data)
        correctPredictions = len([i for i, value in enumerate(predictions) if value == output_data[i]])

        confusionMatrix = tf.math.confusion_matrix(labels=output_data, predictions=predictions).numpy()
        accuracy = correctPredictions / len(predictions)
        loss = None         #TODO: calculate loss
        
        scores = [loss, accuracy]

        self._verbosePrint("Accuracy: {a:.2f}".format(a=accuracy*100))
        self._verbosePrint("Confusion matrix: \n" + str(confusionMatrix))

        self._evaluationHistory = (scores, confusionMatrix)
        return scores, confusionMatrix, predictions


    ## Predict the labels of the input data
    # @param input_data     The input data
    # @return               An array containing the predicted values
    def predict(self, input_data):
        # Get feature vector data
        x = self._getFeatureVectorData(input_data)

        # Predict with the classifier model
        return self._classifier.predict(x)


    ## Store the model to a file for later usage
    # @param paths           Dictoronary with custom paths
    def store(self, paths=None):
        if paths == None:
            paths = self.getPaths(self._path) 

        # Store the model
        super().store(paths)

        # Store the composite classifier
        dump(self._classifier, paths["classifier"])
        self._verbosePrint("Written composite classifier to '{p}'".format(p=paths["classifier"]))


    ## Load a model from file 
    # @param paths           Dictoronary with custom paths
    # @return True or False depending on whether loading succeeded or failed
    def load(self, paths=None):
        if paths == None:
            paths = self.getPaths(self._path) 

        # Check if classifier exists
        if not os.path.isfile(paths["classifier"]):
            self._verbosePrint("File not found: " + paths["classifier"])
            return False
        
        # Load the composite classifier
        self._classifier = load(paths["classifier"])
        if self._classifier == None:
            return False

        self._verbosePrint("Loaded composite classifier from '{p}'".format(p=paths["classifier"]))

        # Load the model
        return super().load(paths)


    ## Get the paths for the model's files
    # @param path       The base path to use
    # @return           Dictionary with paths
    def getPaths(self, path):
        paths = super().getPaths(path)
        paths["classifier"] = os.path.join(path, "composite_classifier.joblib")
        return paths


    ## Visualize the training history
    # @param filepath       Optional parameter; defines where to store the training history plot
    def plotTrainingHistory(self):
        raise Exception("Not available")
