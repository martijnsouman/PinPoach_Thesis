import os
import numpy as np
import json


## DatasetHandler class
#
# Used to load existing datasets
class DatasetHandler:
    _dataset = list()
    _analyzedDataPath = None
    _verboseMode = False
    
    ## Constructor for the DatasetHandler class
    # @param analyzed_data_path         Path to the analyzed data to load
    # @param verbose                    Enable or disable verbose printing
    def __init__(self, analyzed_data_path, verbose=False):
        self._analyzedDataPath = analyzed_data_path

        self._verboseMode = verbose
        self._dataset = list()
   

    ## Load the data from the specified path
    # @return               Boolean; depending on whether loading succeeded or failed
    def loadData(self):
        dataPath = os.path.join(self._analyzedDataPath, "data.npz")
        layerDataPath = os.path.join(self._analyzedDataPath, "layers.json")
    
        # Check if the files exist
        if (not os.path.isfile(dataPath)) or (not os.path.isfile(layerDataPath)):
            return False

        #Load data
        data = np.array(np.load(dataPath)["data"])
        
        #Load layers data
        with open(layerDataPath, "r") as jsonFile:
            layerInfo = json.loads(jsonFile.read())
        
        #Change data format to single list of tuples
        self._dataset = list(zip(data, layerInfo))

        #Shuffle the data
        np.random.shuffle(self._dataset)

        return True



    ## Assign a binary label to the data
    # @param layer_names    The names of the layers to consider as a binary 1
    # @return               xData(input), yData(labels), layerData
    def getBinaryLabeledData(self, layer_names):

        xData, layerData = zip(*self._dataset)
        yData = list()
        for signal in layerData:
            label = 0
            for layer in signal:
                if layer["layer_name"] in layer_names:
                    label = 1
            yData.append(label)
        
        return np.array(xData), np.array(yData), layerData



    ## Split the input and output lists into two separate sets
    # @param x          Input data
    # @param y          Output data
    # @param layer_data Information about the layers
    # @param test_split The percentage of the original data to be put into the test dataset
    # @return testX, testY, trainX, trainY, testLayerData, trainLayerData
    def splitInputAndOutputLists(self, x, y, layer_data, test_split=0.25):
        testSampleSize = int(test_split * len(x))
    
        testInput = x[0:testSampleSize]
        testOutput = y[0:testSampleSize]
        testLayerData = layer_data[0:testSampleSize]

        trainInput = x[testSampleSize:]
        trainOutput = y[testSampleSize:]
        trainLayerData = layer_data[testSampleSize:]

        return testInput, testOutput, trainInput, trainOutput, testLayerData, trainLayerData




    ## Verbose print a string
    # @param string     The string to print
    def _verbosePrint(self, string):
        if(self._verboseMode):
            print("[Dataset Handler] " + str(string))

