from ..genetic import * 

## Class derived from AbstractIndividual to implement a HyperModel individual
class HyperModelIndividual(AbstractIndividual):
    _modelWrapper = None    
    
    ## Constructor for the HyperModelIndividual class
    # @param parameters                 A list of parameters (such as AbstractParameter objects)
    # @param model_wrapper              An instance of a AbstractHyperModel object
    # @param build_function             Callback function to the build model function
    # @param train_function             Callback function to the train model function
    # @param evaluate_function          Callback function to the evaluate model function
    # @param max_model_size             The maximum size of a model. Larger nodels are rejected.
    # @param max_flatten_layer_size     The maximum size of the flatten layer in a model.
    def __init__(self,
            parameters, 
            model_wrapper, 
            build_function,
            train_function,
            evaluate_function,
            max_model_size=None,
            max_flatten_layer_size=None
        ):

        self._modelWrapper = model_wrapper
        self._maxModelSize = max_model_size
        self._maxFlattenSize = max_flatten_layer_size
        self._buildFunction = build_function
        self._trainFunction = train_function
        self._evaluateFunction = evaluate_function

        super().__init__(parameters)


    ## Evaluate the individual and determine the fitness
    def evaluate(self):
        # Clean the model before building, training and evaluation
        self._modelWrapper.clean()

        # Get the hyperparameter values
        values = self.getParameterValuesDict()

        print("Values:")
        for key,value in values.items():
            print("{k}: {v}".format(k=key, v=value))

        # Build the model 
        self._modelWrapper = self._buildFunction(self._modelWrapper, values)

        # Reject the model if it's too large
        numParams = self._modelWrapper.getNumParameters()
        if (self._maxModelSize != None) and (numParams >= self._maxModelSize):
            self._rejectModel("Too many parameters, n={n}".format(n=numParams))
            return

        # Reject the model if the flatten layer is too large
        if self._maxFlattenSize != None:
            flattenLayerSize = self._modelWrapper.getLayer("flatten").output_shape[1:][0]
            if flattenLayerSize >= self._maxFlattenSize:
                self._rejectModel("Flatten layer too large. n={n}".format(n=flattenLayerSize))
                return

        # Train the model
        self._modelWrapper = self._trainFunction(self._modelWrapper)

        # Evaluate the model
        scores, _, _ = self._evaluateFunction(self._modelWrapper)
        accuracy = scores[1]
        
        # Determine fitness
        self._calcFitness(accuracy, numParams)
        print("Fitness: {f} (acc={a})\n".format(f=self._fitness, a=accuracy))
        

    ## Get the model wrapper
    # @return model wrapper
    def getModelWrapper(self):
        return self._modelWrapper



    ## Store the individual to a file 
    # @param base_path       The directory where data is stored 
    def store(self, base_path):
        # Add directory to path 
        base_path = os.path.join(base_path, "individual_{n}".format(n=self._id))
        
        # Store the model
        paths = self._modelWrapper.getPaths(base_path)
        self._modelWrapper.store(paths)
        
        # Store the individual data
        super().store(base_path)


    
    ## Load the individual from a file
    # @param base_path           The directory where data is stored 
    # @param unique_id          The unique id of the individual in this generation
    def load(self, base_path=None, unique_id=None):
        if unique_id != None:
            self._id = unique_id
        
        # Change the base path if base path is set
        if base_path != None:
            # Add directory to path 
            self._path = os.path.join(base_path, "individual_{n}".format(n=unique_id))

        base_path = self._path
        
        # Load the model
        paths = self._modelWrapper.getPaths(base_path)
        if not self._modelWrapper.load(paths):
            raise Exception("Could not load model.")

        # Load the individual data
        super().load(base_path, unique_id)



    ## Reject the model with a description 
    # @param description        Describe why the model has been rejected
    def _rejectModel(self, description):
        print("Model rejected: {d}\n".format(d=description))
        self._fitness = 0.0


    ## Calculate fitness value for this individual
    # @param accuracy       The final accuracy value of the model
    # @param model_size     The number of parameters in the model
    def _calcFitness(self, accuracy, model_size):
        self._fitness = accuracy 
