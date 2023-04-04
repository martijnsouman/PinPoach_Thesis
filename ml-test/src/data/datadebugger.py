import os

from matplotlib import pyplot as plt
from matplotlib import cm

import numpy as np
import array
import math

from pydub import AudioSegment

## DataDebugger class
#
# Used to debug generated data
class DataDebugger:
    _plotOutputPath = None
    _wavOutputPath = None
        

    ## Constructor for the DataDebugger class
    # @param plot_output_path       Path to the stored plot images
    # @param wav_output_path        Path to the stored wav files
    # @param store_dpi              The dpi value for the stored images
    def __init__(self, plot_output_path=None, wav_output_path=None, store_dpi=100):
        self._plotOutputPath = plot_output_path
        self._wavOutputPath = wav_output_path
        self._dpi = store_dpi



    ## Plot a signal
    # @param signal         The signal to plot
    # @param title          The title of the plot
    # @param filename       The name of the file to store
    def plotSignal(self, signal, title="Audio Sample", filename=None):
        fig = plt.figure()
        plt.plot(signal)
        plt.xlabel("Samples")
        plt.title(title)
        self._showOrStoreFigure(fig, filename)
    

    ## Plot a stack of signals
    # @param stacked_signal         The stacked signal
    # @param signals                The signals used in the stack
    # @param layers                 The layer information dictionaries for the stack
    # @param filename               The name of the file to store
    def plotSignalStack(self, stacked_signal, signals, layers, title="", filename=None):
        #The format of the label for each layer
        labelFormat = "{name}:\n{path}\nSamples shifted: {timeshift}\nAmplitude scale: {amp_scale:.2f}"
        
        #Color map
        cmap = plt.cm.get_cmap("hsv", 2*len(layers))

        fig = plt.figure()
        yticks = ["Signal stack"]
        stack = np.interp(stacked_signal, (min(stacked_signal), max(stacked_signal)), (-0.5,0.5))
        plt.plot(stack)

        for i, layer in enumerate(layers):
            #Format the label
            layerLabel = labelFormat.format(
                name=layer["layer_name"],
                path=os.path.split(layer["filepath"])[1],
                timeshift=layer["timeshift"],
                amp_scale=layer["amplitude_scale"])
            
            #Set the name for this y tick
            #yticks.append(layer["layer_name"])
            yticks.append(layerLabel)

            #Normalize / scale the signal
            s = np.interp(signals[i][0], (min(signals[i][0]), max(signals[i][0])), (-0.5,0.5))
            
            #Append a y bias to the signal
            y = s + (i+1)
            
            #Plot the signal
            plt.plot(y, label=layerLabel)

        #cells = list()
        #for layer in layers:
        #    cells.append([layer["filepath"], layer["timeshift"], "{:.2f}".format(layer["amplitude_scale"])])

        #table = plt.table(
        #        cellText=cells, 
        #        rowLabels=[x["layer_name"] for x in layers],
        #        colLabels=["Filename", "Samples shifted", "Amplitude scale"],
        #        loc="bottom", bbox=[0.2, -0.45, 0.8, .28])

        #table.auto_set_font_size(False)
        #table.set_fontsize(7)
        #table.auto_set_column_width(col=list(range(len(cells[0]))))
        #plt.subplots_adjust(bottom=0.3)

        plt.yticks(np.arange(len(yticks)),yticks)
        plt.xlabel("Samples") 
        plt.ylabel("Layers") 
        plt.title(title)
        self._showOrStoreFigure(fig, filename)


    ## Plot MFCCs matrix
    # @param mfccs          MFCCs matrix to plot
    # @param title          The title of the plot
    # @param filename       The name of the file to store
    def plotMFCCs(self, mfccs, title="", filename=None):
        fig = plt.figure()
        
        #Prepare the data for a plot
        mfccs = np.squeeze(mfccs).transpose()

        c = plt.imshow(mfccs, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
        #plt.colorbar(c, format="%-2.f dB")
        plt.colorbar(c)
        plt.title(title)
        self._showOrStoreFigure(fig, filename)

    
    ## Export a signal to a file
    # @param signal         The signal to store to a file
    # @param sample_rate    The sample rate used for this signal
    # @param filename       The name of the file to store
    def exportSignal(self, signal, sample_rate, filename):
        #Determine the duration of the signal
        duration = int(len(signal)/sample_rate * 1000)

        #Scale the signal
        signal = signal * 1000

        sound = AudioSegment.silent(duration, sample_rate)
        sound = sound._spawn(array.array(sound.array_type, signal.astype(int)))

        if self._wavOutputPath == None:
            raise Exception("Wav output path not set.")
        else:
            sound.export(os.path.join(self._wavOutputPath, filename), format="wav")

    
    ## Get the number of occurences for each layer name
    # @param predictions        The predictions made by the machine learning model
    # @return                   A dictionary containing layer names with the occurence count of each layer name
    def getLayerOccurenceCount(self, predictions):
        layerNames = dict()
        for i, prediction in enumerate(predictions):
            for layer in prediction["layer_data"]:
                layerName = layer["layer_name"]

                if layerName in layerNames:
                    layerNames[layerName] += 1
                else:
                    layerNames[layerName] = 1
        
        layerNames["total"] = len(predictions)
        return layerNames
  


    ## Get the number of occurences for each permutation
    # @param predictions        The predictions made by the machine learning model
    # @return                   A dictionary containing layer permutations with the occurence count of each permutation 
    def getLayerPermutationCount(self, predictions):
        layerPermutations = dict()
        for i, prediction in enumerate(predictions):
            layerNames = [x["layer_name"] for x in prediction["layer_data"]]
            permutationName = "_".join(layerNames)
            
            if permutationName in layerPermutations:
                layerPermutations[permutationName] += 1
            else:
                layerPermutations[permutationName] = 1

        return layerPermutations


    ## Get the combination of layers for the predictions
    # @param predictions        The predictions made by the machine learning model
    # @return                   The stacks found in the predictions
    def getLayerStacks(self, predictions):
        stacks = list()
        for i, prediction in enumerate(predictions):
            layers = [x["layer_name"] for x in prediction["layer_data"]]
            stacks.append(layers)

        return stacks
   


    ## Plot the correlations between the predictions and layers in a scatter plot
    # @param float_predictions      The raw floating value output from the ML algorithm
    # @param labels                 The true labels
    # @param layer_data             List with dictionaries, describing the layers in a signal
    # @param class_names            The names of the classes to be classified as a binary 1
    def plotCorrelations(self, float_predictions, labels, layer_data, class_names):
        # Calculate the confidence of prediction (x)
        # How confident was the algorithm of its decision? 0 = Not confident at all / random guess, 1 = Very confident 
        f_confidence = lambda x : 2 * math.fabs(x - 0.5)

        # Calculate the prediction error of a prediction (x) based on the correct value (c)
        # What is the prediction error: 1 = Wrong prediciton 0.5 = random predicted, 0 = perfect prediction
        f_error = lambda x,c : math.fabs(x - c)

        # Determine the predicted label based on prediction(x)
        f_prediction = lambda x : int(x > 0.5)
    

        # Calculate for all predictions
        confidenceValues = [f_confidence(x) for x in float_predictions]
        predictionErrors = [f_error(x, labels[i]) for i, x in enumerate(float_predictions)] 
        predictedLabels = [f_prediction(x) for x in float_predictions]
        correctPredictions = [int(x == labels[i]) for i, x in enumerate(predictedLabels)]

    
        # Find correlation between stack combination and Prediction error / accuracy
        getStackName = lambda i : ",".join([x["layer_name"] for x in layer_data[i] if x["layer_name"] not in class_names])

        # Find all layer combinations
        layerCombinations = list()
        for i, signalLayer in enumerate(layer_data):
            stackName = getStackName(i)
            if stackName not in layerCombinations:
                layerCombinations.append(stackName)


        # Get the error value and correct prediction value for each stack combination
        stackData = dict()
        for i in range(0, len(labels)):
            data = (correctPredictions[i], predictionErrors[i], confidenceValues[i])
            stackName = getStackName(i)
            if stackName not in stackData.keys():
                stackData[stackName] = list()

            stackData[getStackName(i)].append(data)



        # Create dictionaries        
        stackAccuracy = dict.fromkeys(layerCombinations, 0)
        stackErrorValues = dict.fromkeys(layerCombinations, [])
        stackConfidenceValues = dict.fromkeys(layerCombinations, [])
        
        # Place the values in the corresponding dictionary
        for key, value in stackData.items():
            predictionResults, errors, confidences = zip(*value)

            #Calculate the amount of correct values
            pCorrect = len([x for x in predictionResults if x == 1])
            
            #Put the results in the corresponding dictionary 
            stackAccuracy[key] = pCorrect/len(predictionResults)
            stackErrorValues[key] = errors
            stackConfidenceValues[key] = confidences 
        
   

        # Plot the stack accuracy
        acc_x, acc_y = zip(*sorted(zip(stackAccuracy.values(), stackAccuracy.keys())))
        fig = plt.figure()
        plt.title("Prediction accuracy vs layer stack")
        plt.xlabel("Mean prediction accuracy")
        plt.ylabel("Stack name")
        plt.barh(acc_y, acc_x, align='center')
        #plt.scatter(acc_x, acc_y)
        self._showOrStoreFigure(fig, "prediction_accuracy-vs-layer_stack.png")


        


        # Plot the stack error values
        # Calculate the mean error for each stack
        stackErrorMeans = [np.mean(x) for x in stackErrorValues.values()]

        # Sort the error values on mean 
        stackErrorMeans, stackNames, stackErrorValues = zip(*sorted(zip(stackErrorMeans, stackErrorValues.keys(), stackErrorValues.values()), reverse=True))

        # X axis (boxplot)  = error range per layer stack sorted on mean 
        # Y axis labels     = stackNames (layerCombinations) & accuracy
        x = stackErrorValues
        y = ["{name}\nAccuracy: {a:.2f}".format(
            name=key,
            a=stackAccuracy[key] * 100)
            for n, key in enumerate(stackNames)]
        
        # Plot the values 
        fig = plt.figure()
        plt.boxplot(x, vert=0)
        plt.yticks(list(range(1, len(stackNames)+1)), y, rotation=0)
        plt.title("Prediction error vs layer stack")
        plt.xlabel("Prediction error")
        plt.ylabel("Layer stack")
        self._showOrStoreFigure(fig, "prediction_error-vs-layer_stack.png")





        
        # Plot the stack confidence values
        # Calculate the mean error for each stack
        stackConfidenceMeans = [np.mean(x) for x in stackConfidenceValues.values()]

        # Sort the error values on mean 
        stackConfidenceMeans, stackNames, stackConfidenceValues = zip(*sorted(zip(stackConfidenceMeans, stackConfidenceValues.keys(),
            stackConfidenceValues.values())))

        # X axis (boxplot)  = confidence range per layer stack sorted on mean 
        # Y axis labels     = stackNames (layerCombinations) & accuracy
        x = stackConfidenceValues 
        y = ["{name}\nAccuracy: {a:.2f}".format(
            name=key,
            a=stackAccuracy[key] * 100)
            for n, key in enumerate(stackNames)]
        
        # Plot the values 
        fig = plt.figure()
        plt.boxplot(x, vert=0)
        plt.yticks(list(range(1, len(stackNames)+1)), y, rotation=0)
        plt.title("Prediction confidence vs layer stack")
        plt.xlabel("Prediction confidence")
        plt.ylabel("Layer stack")
        self._showOrStoreFigure(fig, "prediction_confidence-vs-layer_stack.png")
   

    ## Get the moving average 
    # @param x          The x data
    # @param n          The size of the moving average
    # @return           The moving average for all samples in x with 'same' padding
    def _getMovingAverage(self, x, n):
        return np.convolve(x, np.ones(n), 'same') / n

     
    ## Sort x and y based on x
    # @param x          The x data
    # @param y          The y data
    # @return           The sorted x and y arrays
    def _sortXY(self, x, y):
        return zip(*sorted(zip(x, y)))




    ## Visualize or store a figure to a file
    # @param fig        The figure
    # @param filename   The name of the file to store
    def _showOrStoreFigure(self, fig, filename):
        if (filename != None) and (self._plotOutputPath != None):
            filepath = os.path.join(self._plotOutputPath, filename)
            fig.savefig(filepath, dpi=self._dpi, bbox_inches='tight')
        else:
            plt.show()

        plt.close(fig)
