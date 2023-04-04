import os


## SignalLayer class
#
# Used to define a layer of signals with various properties
# A SignalLayer is used by the DataGenerator class and defines the properties
# of all signals in this layer.
class SignalLayer:
    _fileList = list()
    _fileListIndex = 0
    
    ## Constructor for the SignalLayer class
    # @param directory              Path to a directory containing audio files
    # @param fixed_duration         The duration of each signal; a signal is modified if larger or smaller than this value
    # @param timeshift_max          The maximum value in which a signal can be shifted in time
    # @param timeshift_bias         Theamount (or mean value) of time each signal is shifted by
    # @param amplitude_scale        Defines the scaling factor for the amplitude for each signal
    # @param amplitude_deviation    The maximum deviation for the amplitude of each signal
    def __init__(self, directory, fixed_duration=1, 
            timeshift_max=0, timeshift_bias=0, 
            amplitude_scale=1, amplitude_deviation=0):

        self._directory = directory
        self._fixedDuration = fixed_duration

        self._timeshiftMax = timeshift_max
        self._timeshiftBias = timeshift_bias

        self._amplitudeScale = amplitude_scale
        self._amplitudeDeviation = amplitude_deviation

        if (amplitude_deviation != None) and ((amplitude_scale - amplitude_deviation) <= 0):
            raise Exception("'amplitude_deviation' must be smaller than 'amplitude_scale'")
        
        #Read all files in directory
        self._fileList = [os.path.join(self._directory, x) for x in os.listdir(self._directory) if x.endswith('.wav')]
        if len(self._fileList) == 0:
            raise Exception("No .wav files in " + str(self._directory))
   
    ## Get the list of files for this layer
    # @return The list of files for this layer
    def getFiles(self):
        return self._fileList
    
    ## Get a file from the layer's directory
    # @return Path to a file
    def takeFile(self):
        filepath = self._fileList[self._fileListIndex]
        self._fileListIndex += 1
        if self._fileListIndex == len(self._fileList):
            self._fileListIndex = 0

        return filepath

    ## Python magic method for casting this class to a string
    # @return   A string containing the name of this layer based on the defined path 
    def __str__(self):
        return os.path.split(self._directory)[1]
    
    ## Get the fixed duration value
    # @return Fixed duration value
    def getFixedDuration(self):
        return self._fixedDuration

    ## Get the maximum timeshift value
    # @return Maximum timeshift value
    def getTimeshiftMax(self):
        return self._timeshiftMax
    
    ## Get the timeshift bias value
    # @param Timeshift bias value
    def getTimeshiftBias(self):
        return self._timeshiftBias
    
    ## Get the amplitude scale value
    # @param Amplitude scale value
    def getAmplitudeScale(self):
        return self._amplitudeScale
    
    ## Get the amplitude deviation value
    # @return Amplitude deviation value
    def getAmplitudeDeviation(self):
        return self._amplitudeDeviation

    def toString(self):
        _config =  str(self.__str__())+\
        "= SignalLayer(\ndirectory="+ str(self._directory)+\
        "\nfixed_duration="+ str(self.getFixedDuration())+\
        "\ntimeshift_bias="+ str(self.getTimeshiftBias())+\
        "\ntimeshift_max="+ str(self.getTimeshiftMax())+\
        "\namplitude_scale="+ str(self.getAmplitudeScale())+\
        "\namplitude_deviation="+ str(self.getAmplitudeDeviation())+\
        "\n)\n"
        return _config
