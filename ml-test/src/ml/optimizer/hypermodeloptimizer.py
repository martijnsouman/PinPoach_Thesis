from .genetic import *
from .individuals import *
from .tuners import *
from ..abstractmodels import *

import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperParameters

import os

## HyperModelOptimizer class
#
# Used to optimize the parameters of inherited AbstractHyperModel classes
class HyperModelOptimizer:
    _modelWrapper = None
    _path = None
    _trainX = None
    _trainY = None
    _testX = None
    _testY = None
    _epochCount = None
    _valSplit = None
    _epochsPatience = None
    _batchSize = None
    _seed = None
    _verboseMode = None
    
    ## Constructor for HyperModelOptimizer class
    # @param model                      An AbstractHyperModel object, to be optimized
    # @param path                       The path to store the optimization files
    # @param train_x                    The input training set
    # @param train_y                    The output training set
    # @param test_x                     The input test set
    # @param test_y                     The output test set
    # @param max_model_size             The maximum size of a model. Larger nodels are rejected.
    # @param max_flatten_layer_size     The maximum size of the flatten layer in a model.
    # Only used when optimizing composite classifiers
    # @param epoch_count                The amount of epochs to train the model
    # @param val_split                  The percentage of which the train data is 
    # split into a separate validation set
    # @param epochs_patience            If the accuracy does not improve after this 
    # many epochs, the training will be stopped
    # @param batch_size                 The size of a training batch
    # @param fixed_seed                 The seed for the random generator
    # @param verbose                    Enable or disable verbose printing
    def __init__(
        self,
        model,
        path,
        train_x, train_y,
        test_x, test_y,
        max_model_size=None,
        max_flatten_layer_size=None,
        epoch_count=100,
        val_split=0.1, 
        epochs_patience=10,
        batch_size=32,
        fixed_seed=1,
        verbose=False
    ):

        self._modelWrapper = model
        self._path = path
        self._trainX = train_x
        self._trainY = train_y
        self._testX = test_x
        self._testY = test_y
        self._maxModelSize = max_model_size
        self._maxFlattenSize = max_flatten_layer_size
        self._epochCount = epoch_count
        self._valSplit = val_split
        self._epochsPatience = epochs_patience
        self._batchSize = batch_size
        self._verboseMode = verbose
    
        # Set the fixed seed
        self._seed = fixed_seed
        tf.random.set_seed(fixed_seed)
        np.random.seed(fixed_seed)
        random.seed(fixed_seed)

        # Check if the model can be optimized 
        if not isinstance(self._modelWrapper, AbstractHyperModel):
            raise Exception("Model is not a AbstractHyperModel instance. This model cannot be optimized.")

        # Create directory
        try:
            os.mkdir(self._path)
        except:
            pass


    ## Optimize with genetic algorithm
    # @param random_first_generation    Use random parameters for the first generation
    # @param generations                The amount of generations for the genetic algorithm
    # @param population_size            The size of one population for the genetic algorithm
    # @param breed_best                 The number of individuals in the population which 
    # are considered as best are bred
    # @param breed_random               The number of individuals in the population are
    # randomly bred with the best
    # @param mutation_deviation         Defines the mutation deviation for each parameter
    # @param individual_class           The class to use for instantiating the individual
    def evolve(self,   
            random_first_generation=True,
            generations=5, 
            population_size=20, 
            breed_best=5,
            breed_random=5,
            mutation_deviation=1,
            individual_class=HyperModelIndividual
        ):

        self._verbosePrint("Optimizing model with genetic algorithm..")
        
        # Create the genetic algorithm object
        geneticAlg = GeneticAlgorithm(self._path, generations, population_size, breed_best, breed_random, mutation_deviation, self._seed)

        # Setup parameters for the inidividual 
        parameters = self._modelWrapper.getHyperParameters()

        # Create an individual
        individual = individual_class(
            parameters,
            self._modelWrapper,
            self._buildCallback, 
            self._trainCallback,
            self._evaluateCallback,
            self._maxModelSize,
            self._maxFlattenSize
        )

        # Find the best parameters
        best = geneticAlg.search(individual, random_first_generation)
        self._optimizedParameters = best.getParameterValuesDict()

        self._verbosePrint("Best fitness: {fitness}".format(fitness=best.getFitness()))
        self._verbosePrint("Values:")
        self._verbosePrint(self._optimizedParameters)
    
        # Store the best model
        best.load()
        self._modelWrapper = best.getModelWrapper()

        self._verbosePrint("Optimization done.")
        self._storeOptimizedParameters() 


    
    ## Optimize the model with keras
    # @param trials                 The amount of trials to run
    # @param use_custom_tuner       Should the model always be optimized with the custom tuner
    def optimize(self, trials=25, use_custom_tuner=False):
        self._verbosePrint("Optimizing hypermodel..")
        
        hp = HyperParameters()
        
        # If the model is a composite classifier
        if isinstance(self._modelWrapper, AbstractCompositeClassifierHyperModel) or use_custom_tuner:
            # Define the Oracle
            customOracle = kt.oracles.BayesianOptimization(
                objective=kt.Objective("test_accuracy", direction="max"),
                max_trials=trials,
                hyperparameters=hp,
                tune_new_entries=True,
                seed=self._seed
            )

            # Define the custom Tuner
            tuner = CustomTuner(
                oracle=customOracle,
                hypermodel=self._modelWrapper,
                max_flatten_layer_size=self._maxFlattenSize,
                max_model_size=self._maxModelSize,
                directory=self._path,
                project_name=self._modelWrapper.getName()
            )
        else:
            # Define the normal Tuner
            tuner = kt.BayesianOptimization(
                hypermodel=self._modelWrapper,
                max_model_size=self._maxModelSize,
                directory=self._path,
                project_name=self._modelWrapper.getName(),
            
                # Oracle parameters
                objective=kt.Objective("val_accuracy", direction="max"),
                max_trials=trials,
                hyperparameters=hp,
                tune_new_entries=True,
                seed=self._seed
            )

        # Print a summary of the search space 
        tuner.search_space_summary()
    
        # Start searching
        es = tf.keras.callbacks.EarlyStopping(patience=self._epochsPatience)
        tuner.search(
            self._trainX, self._trainY, 
            self._testX, self._testY,
            epochs=self._epochCount, validation_split=self._valSplit,
            verbose=1, callbacks=[es], batch_size=self._batchSize
        )
        
        # Print a summary of the results
        tuner.results_summary()
        self._verbosePrint("Optimization done.")
        
        # Get the best model 
        bestModel = tuner.get_best_models(num_models=1)[0]
        if isinstance(bestModel, AbstractHyperModel):
            # The model is a abstract hypermodel, update the modelwrapper
            self._modelWrapper = bestModel
        else:
            # The model is a keras model, update the kerasmodel in the modelwrapper
            self._modelWrapper.setModel(bestModel)
        
        # Store the best hyperparameters
        self._optimizedParameters = tuner.get_best_hyperparameters()[0].values
        self._storeOptimizedParameters() 

        # Write the log file
        log = tuner.getLog()
        log.append({"best trial": tuner.get_best_trial_state()})
        self._writeLog(log)

    
    ## Callback function for building the model with a genetic algorithm
    # @param model      An instance of the AbstractHyperModel class
    # @param values     A dictionary containing the hyperparameter values
    # @return           The model after building
    def _buildCallback(self, model, values):
        model.buildModel(values)
        return model

    ## Callback function for training the model with a genetic algorithm
    # @param model      An instance of the AbstractHyperModel class
    # @return           The model after training 
    def _trainCallback(self, model):
        model.train(self._trainX, self._trainY, 
            val_split=self._valSplit, 
            epoch_count=self._epochCount, 
            epochs_patience=self._epochsPatience,
            batch_size=self._batchSize)
        return model
        
    ## Callback function for evaluating the model with a genetic algorithm
    # @param model      An instance of the AbstractHyperModel class
    # @return           The evaluation results 
    def _evaluateCallback(self, model):
       return model.evaluate(self._testX, self._testY)




    ## Get the results of the optimization algorithm
    # @return model, optimized_parameters
    def getResults(self):
        return self._modelWrapper, self._optimizedParameters
    
    ## Store the optimized parameters to a file
    def _storeOptimizedParameters(self):
        self._verbosePrint("Saving results..")

        with open(os.path.join(self._path, "parameters.json"), "w+") as jsonFile:
            json.dump(self._optimizedParameters, jsonFile)

    
    ## Write data to log file
    # @param data       A dictionary with data for the log file
    def _writeLog(self, data):
        path = os.path.join(self._path, "log.json")
        with open(path, "w+") as jsonFile:
            json.dump(data, jsonFile)


    ## Verbose print a string
    # @param string The string to print
    # @param alwaysPrint Always print the string, even if verbose mode is set to False
    # @param error  The string is an error message
    def _verbosePrint(self, string, alwaysPrint=False, error=False):
        if error:
            string = "ERROR: " + string

        if(self._verboseMode or alwaysPrint or error):
            print("[Hypermodel optimizer] " + str(string))

