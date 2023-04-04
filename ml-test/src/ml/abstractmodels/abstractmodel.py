import os
import json

import numpy as np
import tensorflow as tf

from tensorflow import keras

import tensorflow.python.keras.utils
from keras.optimizers import *

from keras.models import *
import keras.layers as kl

import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt


## An abstract class for a machine learning model
#
# A wrapper class for keras model objects
class AbstractModel:
    _verboseMode = False
    _model = None
    _inputShape = None
    _modelPath = None
    _weightsPath = None
    _name = None
    
    #History variables
    _trainHistory = None
    _evaluationHistory = None
    

    ## Constructor for the AbstractModel class
    # @param path           Path to a directory to load & store the model in
    # @param name           The name of the model
    # @param input_shape    The input data (train x) shape
    # @param verbose        Enable or disable verbose printing
    def __init__(self, path, name, input_shape, verbose=False):
        self._path = os.path.join(path, name) 

        self._inputShape = input_shape

        self._verboseMode = verbose
        self._name = str(name)

   
    ## Get the name of the model
    # @return The name of the model
    def getName(self):
        return self._name
    
    ## Abstract function for building the model 
    def build(self):
        self._verbosePrint("Building model..")

    ## Compile the model
    # @param optimizer_name         The name of the optimizer to use
    # @param learning_rate          The learning rate to use
    def _compile(self, optimizer_name='sgd', learning_rate=0.01):
        self._verbosePrint("Compiling model..")

        if optimizer_name == "sgd":
            opt = SGD(learning_rate=learning_rate)
        elif optimizer_name == "adam":
            opt = Adam(learning_rate=learning_rate)
        elif optimizer_name == "rmsprop":
            opt = RMSprop(learning_rate=learning_rate)
        elif optimizer_name == "adadelta":
            opt = Adadelta(learning_rate=learning_rate)
        elif optimizer_name == "adagrad":
            opt = Adagrad(learning_rate=learning_rate)
        elif optimizer_name == "adamax":
            opt = Adamax(learning_rate=learning_rate)
        elif optimizer_name == "nadam":
            opt = Nadam(learning_rate=learning_rate)
        elif optimizer_name == "ftrl":
            opt = Ftrl(learning_rate=learning_rate)
        
        #Compile the model
        self._model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        #Print a summary
        print("Input shape: " + str(self._inputShape))
        self._model.summary()
   
    

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
        self._verbosePrint("Training model..")
        self._verbosePrint("Input shape: " + str(np.shape(train_x)))
        self._verbosePrint("Output shape: " + str(np.shape(train_y)))
        
        if callbacks == None:
            es = tf.keras.callbacks.EarlyStopping(patience=epochs_patience)
            callbacks = [es]

        hist = self._model.fit(
            train_x, train_y, 
            epochs=epoch_count, validation_split=val_split,
            verbose=1, callbacks=callbacks, batch_size=batch_size
        )
        self._trainHistory = hist.history



    ## Determine the accuracy of the model for the test dataset
    # @param input_data     The input test set
    # @param output_data     The output test set
    # @return scores, confusionMatrix, predictions list
    def evaluate(self, input_data, output_data):
        self._verbosePrint("Evaluating model..")
        
        # Check if the model has a training history
        #if self._trainHistory == None:
        #    self._verbosePrint("Error: Model must be fitted first.")
        #    return None, None, None

        #Calculate scores
        scores = self._model.evaluate(input_data, output_data)

        #Calculate confusion matrix
        predictions = (self._model.predict(input_data) > 0.5).astype("int32")
        confusionMatrix = tf.math.confusion_matrix(labels=output_data, predictions=predictions).numpy()

        self._verbosePrint(str(self._model.metrics_names[1] + ": " + str(scores[1] * 100)))
        self._verbosePrint("Confusion matrix: \n" + str(confusionMatrix))
        
        self._evaluationHistory = (scores, confusionMatrix)
        return scores, confusionMatrix, predictions


    ## Predict the labels of the input data
    # @param input_data     The input data
    # @return               An array containing the predicted values
    def predict(self, input_data):
        return self._model.predict(input_data)



    ## Get the number of parameters
    # @return               The number of parameters
    def getNumParameters(self):
        return self._model.count_params()
   


    ## Get a specific layer from the model
    # @param name       The name of the layer
    # @return           The layer
    def getLayer(self, name):
        for layer in self._model.layers:
            if layer.name == name:
                return layer
        
        return None



    ## Get the training history
    # @return               Dictionary with training history
    def getTrainHistory(self):
        return self._trainHistory


    ## Get the correct and wrongly predicted items
    # @param predictions        A list of predictions made by the model
    # @param output_data        The expected output values 
    # @param layer_data         A list of dictionaries with data describing each sound layer
    # @return correctPredictions, wrongPredictions
    def debug(self, predictions, output_data, layer_data):
        self._verbosePrint("Debugging model..")

        correctPredictions = list()
        wrongPredictions = list()

        for i, prediction in enumerate(predictions):
            data = {
                "prediction": prediction[0],
                "label": output_data[i],
                "layer_data": layer_data[i]
            }

            if prediction == output_data[i]:
                correctPredictions.append(data)
            else:
                wrongPredictions.append(data)

        return correctPredictions, wrongPredictions
    

    ## Store the model to a file for later usage
    # @param paths           Dictoronary with custom paths
    def store(self, paths=None):
        if paths == None:
            paths = self.getPaths(self._path) 
        
        # Try to create the directory
        try:
            os.mkdir(paths["base_path"])
        except:
            pass

        #Write the model to a json file
        modelJson = self._model.to_json()
        with open(paths["model"], "w+") as jsonFile:
            jsonFile.write(modelJson)

        #Write the weights to a json file
        self._model.save_weights(paths["weights"])
        
        #Verbose message
        self._verbosePrint("Written model to '{mp}' and weights of model to '{wp}'".format(mp=paths["model"], wp=paths["weights"]))



    ## Load a model from file 
    # @param paths           Dictoronary with custom paths
    # @return True or False depending on whether loading succeeded or failed
    def load(self, paths=None):
        if paths == None:
            paths = self.getPaths(self._path) 

        #Check if file exists
        if not os.path.isfile(paths["model"]):
            self._verbosePrint("File not found: " + paths["model"])
            return False

        elif not os.path.isfile(paths["weights"]):
            self._verbosePrint("File not found: " + paths["weights"])
            return False

        #Proceed to load the model
        jsonFile = open(paths["model"], "r")
        self._model = model_from_json(jsonFile.read())
        jsonFile.close()

        #Load weights
        self._model.load_weights(paths["weights"])

        #Verbose message
        self._verbosePrint("Loaded model file '{mp}' and weights file '{wp}'".format(mp=paths["model"], wp=paths["weights"]))
        
        #Compile the model
        self._compile()
        
        return True


    
    ## Get the paths for the model's files
    # @param path       The base path to use
    # @return           Dictionary with paths
    def getPaths(self, path):
        return {
            "base_path": path,
            "model": os.path.join(path, "model.json"),
            "weights": os.path.join(path, "weights.h5")
        }


    ## Visualize the model
    # @param filepath       The path to store the graph to
    def plot(self, filepath):
        self._verbosePrint("Saving model graph to " + str(filepath))

        #Visualize the model 
        keras.utils.plot_model(
            self._model,
            to_file=filepath,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )



    ## Visualize the confusion matrix
    # @param labels     The labels for the confusion matrix
    def plotConfusionMatrix(self, labels, path):
        if self._evaluationHistory == None:
            self._verbosePrint("Cannot plot confusion matrix: evaluate the model first.", error=True)
            return
            
        self._verbosePrint("Plotting confusion matrix")
        matrix = self._evaluationHistory[1]

        #Normalize
        normalizedMatrix = np.around(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], decimals=2)

        #Show dataframe
        dataframe = pd.DataFrame(normalizedMatrix, index=labels, columns=labels)

        #Add heatmap
        sn.heatmap(dataframe, annot=True, annot_kws={"size": 32})
        
        #Show plot
        plt.title("Confusion matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(path)
        plt.clf()
        # plt.show()

    
    ## Visualize the training history
    # @param filepath       Optional parameter; defines where to store the training history plot
    def plotTrainingHistory(self, filepath=None):
        if self._trainHistory == None:
            self._verbosePrint("Cannot plot training history: fit the model first.", error=True)
            return
            
        self._verbosePrint("Plotting training history")

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        # fig.suptitle("Training history")
        
        ax1.plot(self._trainHistory['accuracy'], label="Train")
        ax1.plot(self._trainHistory['val_accuracy'], label="Validation")
        ax1.set_title("Model accuracy")
        ax1.set_ylabel("Accuracy")
        # ax1.set_xlabel("Epoch")
        ax1.legend(loc='lower right')

        ax2.plot(self._trainHistory['loss'], label="Train")
        ax2.plot(self._trainHistory['val_loss'], label="Validation")
        ax2.set_title("Model loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(loc='upper right')
        
        if filepath != None:
            fig.savefig(filepath, dpi=400)

        # plt.show()
        plt.close(fig)


    ## Verbose print a string
    # @param string The string to print
    # @param alwaysPrint Always print the string, even if verbose mode is set to False
    # @param error  The string is an error message
    def _verbosePrint(self, string, alwaysPrint=False, error=False):
        if error:
            string = "ERROR: " + string

        if(self._verboseMode or alwaysPrint or error):
            print("[Model: " + self._name + "] " + str(string))

