from data import *
from ml.optimizer import *
from ml.models import *
from ml.abstractmodels import * 
from libhelpers import *
import os

experimentName = "Test8_n2"

#Paths
datasetOutputPath = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/Dataset/Test8_n2"
modelsPath = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/ml-test/src/models/" + experimentName
plotOutputPath = modelsPath + "/plots/"   # For more insights into what predictions were wrong
optimizedOutputPath = modelsPath + "/optimized_output/" 
waveOutputPath = modelsPath + "/samples/"


#Variables
verbose = True
seed = 1
nSize = 2 # number of tracks from the each background noise, 
# to creat a dataset 20 is OK, bu 100 (or even 200) is for better predictions
# If 20, it would create 20 tracks with gunshot and 20 tracks without gunshot 
# for each background noise. (np^2)

oneDimensionalSignals = True

labels = {
    "no shot": 0,
    "shot": 1
}


dataGenerator = DataGenerator(datasetOutputPath, nSize,
        generate_averaged_signals=oneDimensionalSignals, verbose=verbose,
        signal_params={"average_length": 24})
dataHandler = DatasetHandler(datasetOutputPath, verbose=verbose)

def createDirectory(path):
    try: 
        os.mkdir(path)
        return True
    except FileExistsError:
        return False

def loadDataset():
    if not dataHandler.loadData():
        #Layers
        bg_0 = SignalLayer(  # Savanna day and night are the "empty" signals, because it is never silent
            directory = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/SplitFiles/african_savanna_day", # Hard code this
            fixed_duration=10,
            amplitude_scale=1,  # This is now linear
            amplitude_deviation=0.5
        )

        bg_1 = SignalLayer(
            directory = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/SplitFiles/african_savanna_night", # and this
            fixed_duration=10,
            amplitude_scale=1,
            amplitude_deviation=0.5
        )

        rain = SignalLayer(
            directory="C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/SplitFiles/rain",
            fixed_duration=10,
            timeshift_max=5,  
            amplitude_scale=1,
            amplitude_deviation=0.5
        )

        thunder = SignalLayer(
            directory="C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/SplitFiles/thunder",
            fixed_duration=10,
            timeshift_max=5,
            amplitude_scale=1,
            amplitude_deviation=0.5
        )

        single_shots = SignalLayer(  # 50% will have a gunshot and the other half won't
            directory="C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/SplitFiles/single_shots",
            fixed_duration=10,
            timeshift_bias=0,
            timeshift_max=3.2,
            amplitude_scale=3,  # signal amplitude 
            amplitude_deviation=2.5  # this creates range of scale +- deviation
        )
        
        # Define the options for the noise layers
        noiseLayerOptions = [
            [bg_0, bg_1],
            [thunder, None],
            [rain, None]
        ]
        
        # Set the layers
        dataGenerator.setNoiseOptions(noiseLayerOptions)
        dataGenerator.setLabeledLayerOptions([single_shots])
        
        # Generate all data
        dataGenerator.generateAll()

        # Try to load the data again
        dataHandler.loadData()

def reconstructPredictions(predictions,categoryName):

    predictedCategory  = waveOutputPath + categoryName
    createDirectory(predictedCategory)    
    predictionDirectory  = predictedCategory + "/predictions/"
    createDirectory(predictionDirectory)    
    exportSignals  = predictedCategory + "/signals/"
    createDirectory(exportSignals)    
    stackDirectory  = predictionDirectory + "/stacks/"
    createDirectory(stackDirectory) 
    # mfccDirectory  = predictionDirectory + "/mfccs/" 
    # createDirectory(mfccDirectory) 
    dataDebugger = DataDebugger(predictedCategory, exportSignals, store_dpi=250)

    for i, prediction in enumerate(predictions):
        layerData = prediction["layer_data"]
        trueLabel = prediction["label"]
        predictedLabel = prediction["prediction"]
        correctPrediction = int(trueLabel == predictedLabel)
        
        if(predictedLabel==1):

            #Reconstruct the different signals, the stack of the signals and the mfccs of the stack
            lSamples, eSamples, sSample, mfccs = dataGenerator.reconstructSignal(layerData)
            
            #Determine a file name and description
            filename = str(i) + "_" + str(correctPrediction) + "_" + str(trueLabel) + "_" + str(predictedLabel)
            description = "True label='" + str(trueLabel) + "', predicted label='" + str(predictedLabel) + "', prediction correct=" + str(correctPrediction)
            
            #Plot the signal and mfccs
            dataDebugger.plotSignalStack(sSample, eSamples, layerData, description, "predictions/stacks/" + filename + ".png")
            # dataDebugger.plotMFCCs(mfccs, description, "predictions/mfccs/" + filename + ".png")

            #Export the signal stack
            # dataDebugger.exportSignal(sSample, 48000, filename + ".wav")

def evaluateModel(model, xTest, yTest, layerTest, name):
    
    # Test the machine learning algorithm
    _, _, predictions = model.evaluate(xTest, yTest)
    pCorrect, pWrong = model.debug(predictions, yTest, layerTest)

    # Reconstruct all wrong predicted audio files
    # reconstructPredictions(pWrong,"pWrong")
    
    # Plot the confusion matrix 
    model.plotConfusionMatrix(labels.keys(), os.path.join(plotOutputPath + "/conf_mtx-"+ name + ".png"))

    # Plot the model
    model.plot(os.path.join(plotOutputPath, model.getName() + "-layout.png"))
    
    # Plot the convolution kernels if the model is inherited from AbstractConvolutionalModel 
    # if isinstance(model, AbstractConvolutionalModel):
        # model.plotConvolutionKernels()

def findCorrelations(model, xTest, yTest, layerTest):
    predictions = model.predict(xTest)
    dataDebugger.plotCorrelations(predictions, yTest, layerTest, ["single_shots"])

def main():
    # Setup libraries 
    setUseGPU(False)    # You can change this
    setSeed(seed)

    # Create directories
    createDirectory(datasetOutputPath)
    createDirectory(modelsPath)
    createDirectory(plotOutputPath)
    createDirectory(optimizedOutputPath)
    # createDirectory(waveOutputPath)
    
    # Load the dataset
    loadDataset()   # THis does everything to create the data

    # Get labeled data
    xData, yData, layerData = dataHandler.getBinaryLabeledData(["single_shots"])
    #print("XData: ", xData)
    print("xData.shape: ", xData.shape)  # 1D=True: (32, 20000, 1), 1D=False: (32, 200, 13, 1)
    print("yData.shape: ", yData.shape)  # Alwasy (32,)
    #print(yData.shape)  # [0 1 0 1 1 0 0 ...] 
    #print("layerData: ", layerData)

    # From here on the training starts

    # Split the dataset
    xTest, yTest, xTrain, yTrain, layerTest, layerTrain = dataHandler.splitInputAndOutputLists(xData, yData, layerData)
    print("xTest.shape: ", xTest.shape)
    print("yTest.shape: ", yTest.shape)
    print("xTrain.shape: ", xTrain.shape)
    print("yTrain.shape: ", yTrain.shape)
    print(type(xTest))
    
    # Training Simple CNN model
    epoch_count = 5
    # Define convolutional layer range
    convRange = range(3, 8)  # (3, 8)
    # Define dense layer range
    denseRange = range(5, 0, -1)  # (5, 0, -1)
    # Build and train all models in specified ranges
    MainConv1DModel(
        modelsPath,
        convRange, 
        denseRange, 
        xTrain, 
        yTrain, 
        xTest, 
        yTest, 
        layerTrain,
        labels,
        epoch_count)



    # Setup the model factory
    #modelFactory = ModelFactory(
    #        modelsPath, 
    #        input_shape=np.shape(xTrain)[1:], 
    #        verbose=verbose)

    # This is a normal CNN model

    # Training a (pre-defined parameters) CNN model
    #hypermodel = modelFactory.getCNNModel(experimentName)
    #if not hypermodel.load():
    #    hp = HyperParameters()  # instantiate parameters
    #    hypermodel.build()  # built model with CNNmodel, this does not work for 1D input
    #    hypermodel.train(xTrain, yTrain, epoch_count=100, epochs_patience=25, batch_size=8)
    #    hypermodel.store()
    #    CNNModel.plotTrainingHistory(hypermodel, os.path.join(plotOutputPath,"training_history.png"))
  
    # Training a HyperCNN 1D model
    #hypermodel = modelFactory.getConv1DHyperModel(experimentName)
    #print(hypermodel)
    
    #if not hypermodel.load():
    #    # Setup the optimizer
    #    opt = HyperModelOptimizer(
    #            hypermodel,
    #            optimizedOutputPath,
    #            xTrain, yTrain,
    #            xTest, yTest,
    #            max_model_size=2000000,
    #            max_flatten_layer_size=2000,
    #            epoch_count=100,
    #            val_split=0.2,
    #            epochs_patience=20,
    #            batch_size=32,
    #            fixed_seed=seed,
    #            verbose=True)  
    #    print(isinstance(opt._modelWrapper, AbstractHyperModel))
    #    print(opt)
    #    # Optimize the model
    #    opt.optimize(trials=100, use_custom_tuner=True)
    #    opt.evolve(generations=5, population_size=20)  # This creates 100 models from which 1 will be the optimal (beste accuracy)
    #    
    #    # Get the results of the optimization steps 
    #    hypermodel, params = opt.getResults()
    #
    #    hypermodel.store()   
         
    # Fancy model

    # Training a HyperCNN model
    #hypermodel = modelFactory.getCNNHyperModel(experimentName)
    #if not hypermodel.load():
    #    # Setup the optimizer
    #    opt = HyperModelOptimizer(
    #            hypermodel,
    #            optimizedOutputPath,
    #            xTrain, yTrain,
    #            xTest, yTest,
    #            max_model_size=2000000,
    #            max_flatten_layer_size=2000,
    #            epoch_count=100,
    #            val_split=0.2,
    #            epochs_patience=20,
    #            batch_size=32,
    #            fixed_seed=seed)

    #    # Optimize the model
    #    # opt.optimize(trials=100, use_custom_tuner=True)
    #    opt.evolve(generations=5, population_size=20)  # This creates 100 models from which 1 will be the optimal (beste accuracy)
        
    #    # Get the results of the optimization steps 
    #    hypermodel, params = opt.getResults()
    
    #    hypermodel.store()    


    # Plot the model and confusion matrix
    #evaluateModel(hypermodel, xTest, yTest, layerTest, experimentName)
        
    # Find correlations between layers and accuracy
    # findCorrelations(hypermodel, xTest, yTest, layerTest)


if __name__ == '__main__':
    main()
