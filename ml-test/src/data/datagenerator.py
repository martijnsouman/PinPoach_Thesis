import os
import json
import array
import time
import random
import copy

from pydub import AudioSegment
import librosa

import tensorflow as tf
import numpy as np

## DataGenerator class
#
# Used to generate augmented signal data, consisting of stacked SignalLayer objects 
class DataGenerator:
    _currentBatchNumber = 0
    
    ## Constructor for the DataGenerator class
    # @param dataset_output_path        Path where all generated files are stored
    # @param batch_size                 The size of one generated batch
    # @param generate_averaged_signals  Should the generated data also contain averaged signals?
    # @param target_sample_rate         The used sample rate 
    # @param target_sample_length       The used sample length
    # @param stft_params                Dictionary with STFT parameters
    # @param mfccs_params               Dictionary with MFCCs parameters
    # @param signal_params              Dictionary with Signal preprocessing parameters
    # @param verbose                    Enable or disable verbose printing
    def __init__(self, dataset_output_path, batch_size,
            generate_averaged_signals=False,
            target_sample_rate=48000, target_sample_length=10,
            stft_params=None, mfccs_params=None, signal_params=None,
            verbose=False):

        self._verboseMode = verbose
        self._datasetOutputPath = dataset_output_path
        self._batchSize = batch_size
        self._generateAveragedSignals = generate_averaged_signals 
        self._targetSampleRate = target_sample_rate
        self._targetSampleLength = target_sample_length * target_sample_rate

        self._stftParams = {
            #Milliseconds per frame
            "frame_ms": 50,

            #Milliseconds per step
            "step_ms": 50,

            #Size of the fft. 
            "fft_size": 1024
        }
        if stft_params != None:
            self._stftParams = stft_params 

        self._mfccsParams = {
            #How many bins are created
            "mel_bins": 26,
            
            #How many mel bins are used for output
            "mel_bins_output": 13,

            #What is the maximum frequency in the MFCCs output
            "max_frequency": 20000.0
        }
        if mfccs_params != None:
            self._mfccsParams = mfccs_params

        self._signalParams = {
            #Average window length
            "average_length": 24,
        }
        if signal_params != None:
            self._signalParams = signal_params


    ## Set the layer options for the noise layers
    #
    # Example usage:
    # @code
    # layerOpts = [(bg0, bg1), (n0, None), (n1, None)]
    # dg = DataGenerator(..)
    # dg.setNoiseOptions(layerOpts)
    # @endcode
    #
    # @param layer_options          A list of tuples containing SignalLayer objects
    def setNoiseOptions(self, layer_options):
        self._noiseLayerOptions = layer_options



    ## Set the labeled layer options for generating the data later
    # These layer options are used for the top signal layer.
    # 
    # The amount of permutations is calculated with p = 2^(n+l)
    # where
    # p is the amount of permutations,
    # n is the number of noise layers,
    # l is the number of labeled layers
    #
    # @param layer_options          A list of SignalLayer objects to be used in the top layer
    def setLabeledLayerOptions(self, layer_options):
        self._labeledLayerOptions = layer_options

    ## Generate data based on the options set by setNoiseOptions and setLabeledLayerOptions
    def generateAll(self):
        layerPermuations = list()
        
        # Determine all layer permutations
        for opt in self._labeledLayerOptions:
            layerOptions = copy.copy(self._noiseLayerOptions)
            layerOptions.append((opt, None))
            
            # Extend the list with permutations
            layerPermuations.extend(self.getLayerPermutations(layerOptions))
        

        # Generate all layers permuations
        for i, layers in enumerate(layerPermuations):
            self._verbosePrint("Batch {i} of {n}.".format(i=i+1, n=len(layerPermuations)))
            self.generate(layers, i)

        # Merge the data files
        self._mergeDataFiles(len(layerPermuations))

        self._verbosePrint("Done.")




       
    ## Determine all the possible permutations given the layer options
    # 
    # layer_options example: [[l1, l0], [None, l2], [l3, None]]
    #
    # @param layer_options          A list of all options for layers 
    # @return                       A list of permuations
    def getLayerPermutations(self, layer_options):
        permutations = list()

        #Calculate the amount of permutations
        nPermutations = 2 ** len(layer_options)
        
        #Loop through all permutations
        for i in range(0, nPermutations):
            perm = list()

            #Loop through the layers
            for j in range(0, len(layer_options)):
                #Select an option
                optionIndex = int(i & (1 << j) > 0)
                option = layer_options[j][optionIndex]
                
                #Append the option if not None
                if option != None:
                    perm.append(option)
            
            #Append the permutation to the list
            permutations.append(perm)
        
        return permutations




    
    
    ## Generate a batch of signals 
    # @param layers         The layers to use for generating this batch
    # @param batch_number   The number of this batch 
    def generate(self, layers, batch_number):
        self._verbosePrint("Generating signal stack: " + str([str(x) for x in layers]))
        startTime = time.time()

        #Parameters shape=(n layers, n samples)
        self._verbosePrint("Generating layer parameters..")
        timeshiftParams, amplitudeParams = self._generateLayerParameters(layers, self._batchSize)


        #Load batch
        #Filebatch shape = (n layers, n samples)
        #sample batch shape = (n layers, n samples, sample size)
        self._verbosePrint("Loading sample batch..")
        fileBatch, sampleBatch = self._loadSampleBatch(layers, self._batchSize)


        #Edit samples
        #Batch shape = (n layers, n samples, sample size)
        self._verbosePrint("Editing sample batch..")
        sampleBatch = self._editSampleBatch(sampleBatch, timeshiftParams, amplitudeParams)


        #Stack the samples
        self._verbosePrint("Stacking samples..")
        stackedSamples = self._stackSampleBatch(sampleBatch, len(layers), self._batchSize)

        # Check which data to use
        if self._generateAveragedSignals:
            #Preprocess the signals
            self._verbosePrint("Preprocessing signals..")
            data = self._preprocessRawSignals(stackedSamples)
        else:
            #Extract MFCCs for all samples
            self._verbosePrint("Extracting MFCCs..")
            data = self._extractMFCCs(stackedSamples)
        
    
        #Export the data to a file
        self._verbosePrint("Exporting data..")
        layerNames = [str(x) for x in layers]
        self._exportData(self._batchSize, batch_number, layerNames, timeshiftParams, amplitudeParams, fileBatch, data)

        #Print the duration of generating this batch
        self._verbosePrint("Done generating batch. Duration: {duration:.2f} seconds".format(duration=(time.time()-startTime)))



    ## Reconstruct a signal based on layer_data
    # @param layer_data             The data for the signal to reconstruct
    # @return                       layerSamples, editedSamples, stackedSamples, mfccs
    def reconstructSignal(self, layer_data):
        self._verbosePrint("Generating single signal stack from layer data..")
        
        #Create placeholder array and lists
        layerSamples = np.zeros(shape=(len(layer_data), 1, self._targetSampleLength), dtype=np.float32)
        layerTimeshiftParams = list()
        layerAmplitudeParams = list()
        
        #Get the sample values
        for i, layer in enumerate(layer_data):
            layerSamples[i][0] = self._loadSample(layer["filepath"])
            layerTimeshiftParams.append([layer["timeshift"]])
            layerAmplitudeParams.append([layer["amplitude_scale"]])
        
        #Edit the samples
        editedSamples = self._editSampleBatch(layerSamples, layerTimeshiftParams, layerAmplitudeParams)
        
        #Stack the samples
        stackedSamples = self._stackSampleBatch(editedSamples, len(layer_data), 1)
        
        #Calculate MFCCs
        mfccs = self._extractMFCCs(stackedSamples)

        return layerSamples, editedSamples, stackedSamples[0], mfccs
    



        

    ## Generate the parameters for each layer
    # 
    # Ouput shape = timeshiftParameters/amplitudeParameters (n layers, n samples)
    # 
    # @param layers         The layers to use for generating layer parameter values
    # @param n              The amount of items to generate
    # @return               layerTimeshiftParams, layerAmplitudeParams
    def _generateLayerParameters(self, layers, n):
        layerTimeshiftParams = list()
        layerAmplitudeParams = list()

        for layer in layers:
            #Generate timeshift params
            if layer.getTimeshiftMax() == 0:
                timeshiftParams = tf.zeros(shape=(n,), dtype=tf.int32)
            else:
                minVal = int(-layer.getTimeshiftMax() * self._targetSampleRate)
                maxVal = int(layer.getTimeshiftMax() * self._targetSampleRate)
                timeshiftParams = tf.random.uniform(
                    shape=(n,),
                    minval=minVal+layer.getTimeshiftBias(),
                    maxval=maxVal+layer.getTimeshiftBias(),
                    dtype=tf.int32)

            #Generate amplitude params
            amplitudeParams = tf.random.uniform(
                shape=(n,),
                minval=layer.getAmplitudeScale()-layer.getAmplitudeDeviation(),
                maxval=layer.getAmplitudeScale()+layer.getAmplitudeDeviation(),
                dtype=tf.float32)
        
            #Append both parameters to the lists
            layerTimeshiftParams.append(timeshiftParams)
            layerAmplitudeParams.append(amplitudeParams)

        return layerTimeshiftParams, layerAmplitudeParams



    ## Load a batch of n samples from layers
    # 
    # Output shape = files:(n layers, n samples), samples:(n layers, n samples, sample length)
    # 
    # @param layers         The layers where signals are retrieved from 
    # @param n              The batch size 
    # @return               layerFiles, layerSamples
    def _loadSampleBatch(self, layers, n):
        layerSamples = np.zeros(
            shape=(len(layers), n, self._targetSampleLength), dtype=np.float32)

        layerFiles = np.zeros(
            shape=(len(layers), n), dtype=np.object)
        
        #Loop through all layers
        for i, layer in enumerate(layers):
            #Get n samples
            for j in range(0, self._batchSize):
                filepath = layer.takeFile()
                layerSamples[i][j] = self._loadSample(filepath)
                layerFiles[i][j] = filepath

        return layerFiles, layerSamples


    ## Load a single sample file
    # 
    # ouput shape = (sample length,)
    #
    # The signal is converted to mono and the duration of the signal is fixed 
    #
    # @param filepath       Path to the file to load
    # @return               A numpy array of the loaded signal
    def _loadSample(self, filepath):
        s, fs = librosa.load(filepath, sr=self._targetSampleRate, dtype='float32', mono=True)
            
        #Convert to the right size
        sizeDiff = len(s) - self._targetSampleLength
        if sizeDiff > 0:
            s = s[:self._targetSampleLength]
        elif sizeDiff < 0:
            s = np.append(s, np.zeros(shape=-sizeDiff))
        
        return s

    

    ## Edit a batch of samples with timeshift and amplitude parameters
    # @param samples            A list of signal layers containing lists of samples to edit
    # @param timeshift_params   The parameters used for timeshifting
    # @param amplitude_params   The parameters used for amplitude editing
    # @return                   The edited samples
    def _editSampleBatch(self, samples, timeshift_params, amplitude_params):
        newSamples = np.copy(samples)
        for i, layer in enumerate(samples):
            for j, sample in enumerate(layer):
                newSamples[i][j] = self._editSample(newSamples[i][j], timeshift_params[i][j], amplitude_params[i][j])

        return newSamples 



    ## Edit a sample
    #
    # output shape = (sample length,)
    #
    # @param sample     The sample to edit
    # @param timeshift  The amount of time the sample is shifted with
    # @param amplitude  The amplitude scale
    # @return           The edited sample
    def _editSample(self, sample, timeshift, amplitude):
        sample = self._normalizeSample(sample)
        sample = self._timeshiftSample(sample, timeshift)
        sample = self._amplitudeScaleSample(sample, amplitude)

        return sample


    ## Normalize a sample
    #
    # output shape = (sample length,)
    #
    # @param sample     The sample to (amplitude) normalize
    # @return           The normalized sample (numpy array)
    def _normalizeSample(self, sample):
        return np.interp(sample, (min(sample), max(sample)), (-1,1))

       
    ## Timeshift a sample
    #
    # output shape = (sample length,)
    # 
    # @param sample     The sample to apply timeshifting on
    # @param timeshift  The amount of time the sample is shifted with
    # @return           The timeshifted sample
    def _timeshiftSample(self, sample, timeshift):
        if timeshift == 0:
            return sample

        sample = np.roll(sample, shift=timeshift, axis=0)

        #Apply zeros on the shifted range
        if timeshift > 0:
            sample[:timeshift] = 0
        else:
            sample[timeshift:] = 0


        return sample


    ## Scale the amplitude of a sample
    #
    # output shape = (sample length,)
    #
    # @param sample     The sample to apply the amplitude scaling on
    # @param scale      The scaling factor
    # @return           The scaled sample
    def _amplitudeScaleSample(self, sample, scale):
        return sample * scale 
    


    ## Stack a batch of samples
    # 
    # Ouput shape: (n samples, sample length)
    #
    # @param samples        A list(item for each layer) of lists containing samples
    # @param n_layers       The amount of layers to stack
    # @param n_samples      The amount of samples to stack 
    # @return               The stacked samples
    def _stackSampleBatch(self, samples, n_layers, n_samples):
        stackedSamples = np.zeros(shape=(n_samples, self._targetSampleLength))

        for j in range(0, n_samples):
            #Create empty sound
            duration = int(self._targetSampleLength/self._targetSampleRate * 1000)
            stackedSound = AudioSegment.silent(duration, self._targetSampleRate)
            
            #Stack the signals for each layer
            for i in range(0, n_layers):
                scaledSamples = samples[i][j] * 1000

                sound = AudioSegment.silent(duration, self._targetSampleRate)
                sound = sound._spawn(array.array(sound.array_type, scaledSamples.astype(int)))

                #Overlay the new sound
                stackedSound = stackedSound.overlay(sound)
            
            #Set the signal
            stackedSamples[j] = self._normalizeSample(stackedSound.get_array_of_samples())
        
        return stackedSamples



    ## Preprocess the raw signals (take the average value)
    # @param signals        The signals to preprocess
    # @return               Numpy array with averaged values
    def _preprocessRawSignals(self, signals):
        newSampleSize = int(self._targetSampleLength / self._signalParams["average_length"])

        reshapedSignals = signals.reshape(signals.shape[0], self._signalParams["average_length"], newSampleSize)
        averages = np.average(reshapedSignals, axis=1)
        return np.expand_dims(averages, axis=-1)


    
    ## Extract MFCCs from samples
    # @param samples        The samples to extract MFCCs from
    # @return               A numpy array containing extracted MFCCs matrixes
    def _extractMFCCs(self, samples):
        #Convert numpy array to tensor
        samples = tf.convert_to_tensor(samples, dtype=tf.float32)

        winSize = int(self._targetSampleRate * (0.001 * self._stftParams["frame_ms"]))
        stepSize = int(self._targetSampleRate * (0.001 * self._stftParams["step_ms"]))

        #Calculate STFT
        stfts = tf.signal.stft(samples, frame_length=winSize, frame_step=stepSize, fft_length=self._stftParams["fft_size"])
        spectrograms = tf.abs(stfts)

        
        nMelBins = self._mfccsParams["mel_bins"]
        lowerEdgeHz = 0
        upperEdgeHz = self._mfccsParams["max_frequency"]

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            nMelBins, num_spectrogram_bins, self._targetSampleRate, lowerEdgeHz, upperEdgeHz)

        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))



        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log_mel_spectrograms
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :self._mfccsParams["mel_bins_output"]]

        #Expand the dimensions and return
        mfccs = tf.expand_dims(mfccs, axis=-1)
        return mfccs.numpy()


   
    

    ## Export the generated data
    # @param n_samples              The amount of samples
    # @param batch_number           The number of this batch
    # @param layer_names            The names of all layers used in the current batch
    # @param timeshift_parameters   The timeshift values for all samples
    # @param amplitude_parameters   The amplitude scaling parameters for all samples
    # @param file_paths             The paths of all used files
    # @param data                   The data to export
    def _exportData(self, n_samples, batch_number, layer_names, timeshift_parameters, amplitude_parameters, file_paths, data):

        # Determine filepaths
        dataFilepath = os.path.join(self._datasetOutputPath, "data_{n}.npz".format(n=batch_number))
        layersInfoFilepath = os.path.join(self._datasetOutputPath, "layers.json")

        # Store the layer information to a file
        self._appendLayerInformation(layersInfoFilepath, n_samples, layer_names, timeshift_parameters, amplitude_parameters, file_paths)

        # Export the data to a compressed file
        np.savez_compressed(dataFilepath, data=data)






    ## Append layer information to a (possibly) existing file
    # @param filepath               Path to the file
    # @param n_samples              The amount of samples
    # @param layer_names            The names of all layers used in the current batch
    # @param timeshift_parameters   The timeshift values for all samples
    # @param amplitude_parameters   The amplitude scaling parameters for all samples
    # @param file_paths             The paths of all used files
    def _appendLayerInformation(self, filepath, n_samples, layer_names, timeshift_parameters, amplitude_parameters, file_paths):
        #Get the file content
        layerInformation = list()
        if os.path.isfile(filepath):
            with open(filepath, "r") as jsonFile:
                #Read content
                layerInformation = json.loads(jsonFile.read())
        
        #Generate format for layer information
        #Loop through all samples
        for i in range(0, n_samples):
            data = list()

            #Loop through all layers
            for j, name in enumerate(layer_names):
                data.append({
                    "layer_name": name,
                    "filepath": file_paths[j][i],
                    "timeshift": int(timeshift_parameters[j].numpy()[i]),
                    "amplitude_scale": float(amplitude_parameters[j].numpy()[i])
                })
            
            layerInformation.append(data)

        #Store the layer information
        with open(filepath, "w+") as jsonFile:
            #Write the layers data to the file
            json.dump(layerInformation, jsonFile, indent=4)

    
    


    ## Merge all generated datafiles from all batches
    # Stores the merged file to 'data.npz'
    # @param num_files          The number of data files to merge
    def _mergeDataFiles(self, num_files):
        self._verbosePrint("Merging all data files..")

        mergedFilepath = os.path.join(self._datasetOutputPath, "data.npz")
        data = np.array([])
        
        # Loop for all files
        for i in range(0, num_files):
            # Determine the path for this data batch file
            filepath = os.path.join(self._datasetOutputPath, "data_{n}.npz".format(n=i))
            
            if not os.path.isfile(filepath):
                raise Exception("File '{f}' for batch number {n} not found.".format(f=filepath, n=i))
            else:
                # Load and concatenate the data
                loadedData = np.load(filepath)["data"]
                if len(data) == 0:
                    data = loadedData
                else:
                    data = np.concatenate((data, loadedData))

                # File is not needed anymore; delete the file
                os.remove(filepath)


        self._verbosePrint("Storing merged data to file..")

        # Store the final file
        np.savez_compressed(mergedFilepath, data=data)




    

    ## Verbose print a string
    # @param string The string to print
    # @param error  The string is an error message
    # @param indent The amount of tabs to use before printing the string 
    def _verbosePrint(self, string, error=False, indent=0):
        if error:
            string = "ERROR: " + string

        if self._verboseMode or error:
            for x in range(0, indent):
                string = "\t" + string

            print("[Data Generator] " + str(string))
        
