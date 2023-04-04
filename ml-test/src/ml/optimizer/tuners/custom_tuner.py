import kerastuner as kt
import tensorflow as tf
import os


## CustomTuner class derived from keras tuner Tuner class
#
# This tuner is able to optimize custom models derived from AbstractHyperModel
class CustomTuner(kt.Tuner):
    
    ## Constructor for the custom tuner
    # @param oracle                 Instance of Oracle class.
    # @param hypermodel             Instance of AbstractHyperModel class
    # @param max_flatten_layer_size The maximum size of the flatten layer in a model.
    # @param max_model_size         Int. Maximum number of scalars
    # in the parameters of a model. Models larger than this are rejected.
    # @param distribution_strategy   Optional. A TensorFlow
    # `tf.distribute` DistributionStrategy instance. If
    # specified, each trial will run under this scope. For
    # example, `tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])`
    # will run each trial on two GPUs. Currently only
    # single-worker strategies are supported.
    # @param directory              String. Path to the working directory (relative).
    # project_name: Name to use as prefix for files saved by this Tuner.
    # @param logger                 Optional. Instance of Logger class, used 
    # for streaming data to Cloud Service for monitoring.
    # @param tuner_id               Optional. If set, use this value as the id of
    # this Tuner.
    # @param overwrite              Bool, default `False`. If `False`, reloads
    # an existing project of the same name if one is found. Otherwise, overwrites the project.

    def __init__(
        self,
        oracle,
        hypermodel,
        max_flatten_layer_size=None,
        max_model_size=None,
        distribution_strategy=None,
        directory=None,
        project_name=None,
        logger=None,
        tuner_id=None,
        overwrite=False,
    ):
        # Call the parent constructor 
        super().__init__(
            oracle, 
            hypermodel, 
            max_model_size, 
            None, 
            None,
            None,
            distribution_strategy,
            directory,
            project_name,
            logger,
            tuner_id,
            overwrite
        )

        # Store the original model wrapper
        self._modelWrapper = hypermodel

        # Store the max model size
        self._maxModelSize = max_model_size
        self._maxFlattenSize = max_flatten_layer_size

        # Create a log
        self._log = list()


    ## Run a trial to evaluate a set of hyperparameter values
    # @param trial              A `Trial` instance that contains the information
    # needed to run this trial. `Hyperparameters` can be accessed
    # via `trial.hyperparameters`.
    # @param *fit_args          Positional arguments passed by `search`
    # @param *fit_kwargs        Keyword arguments passed by `search`
    def run_trial(self, trial, *fit_args, **fit_kwargs):
        # Obtain the hyperparameters for this trial
        hp = trial.hyperparameters

        # Clean the model
        self._modelWrapper.clean()

        # Create metrics dictionary
        metrics = {
            "accuracy": 0,
            "loss": 0,
            "val_loss": 0,
            "val_accuracy": 0,
            "num_params": 0,
            "test_accuracy": 0
        }
        
        # Build the model
        self._modelWrapper.build(hp)
        
        # Get model properties
        metrics["num_params"] = self._modelWrapper.getNumParameters()
        flattenLayerSize = self._modelWrapper.getLayer("flatten").output_shape[1:][0]

        # Reject the model if it's too large
        if (self._maxModelSize != None) and (metrics["num_params"] >= self._maxModelSize):
            print("Model rejected: Too many parameters.")

        # Reject the model if the flatten layer is too large
        elif (self._maxFlattenSize != None) and (flattenLayerSize >= self._maxFlattenSize):
            print("Model rejected: Flatten layer too large.")

        else:
            callbacks = fit_kwargs['callbacks']

            # Train the model
            self._modelWrapper.train(
                train_x=fit_args[0],
                train_y=fit_args[1],
                epoch_count=fit_kwargs['epochs'],
                val_split=fit_kwargs['validation_split'],
                callbacks=fit_kwargs['callbacks'],
                batch_size=fit_kwargs['batch_size']
            )
            
            # Update the metrics
            for key, value in self._modelWrapper.getTrainHistory().items():
                if type(value) == list:
                    metrics[key] = value[0]
                else:
                    metrics[key] = value
            
            # Evaluate the model
            testScores, _, _ = self._modelWrapper.evaluate(fit_args[2], fit_args[3])
            metrics["test_accuracy"] = testScores[1]


        # Save the model
        self.save_model(trial.trial_id, self._modelWrapper)

        # Update the trial values
        self.oracle.update_trial(trial.trial_id, metrics=metrics, step=0)

        # Update the log
        self._addToLog(trial.trial_id, metrics)


    ## Save the model to a file
    # @param trial_id       The id of the trial
    # @param model_wrapper  The model (instance of AbstractHyperModel) to store to a file
    def save_model(self, trial_id, model_wrapper):
        path = os.path.join(self.get_trial_dir(trial_id), "checkpoint")
        paths = model_wrapper.getPaths(path)
        model_wrapper.store(paths)


    ## Load the model from a file
    # @param trial      The id of the trial
    # @return Instance of AbstractHyperModel
    def load_model(self, trial):
        path = os.path.join(self.get_trial_dir(trial.trial_id), "checkpoint")
        paths = self._modelWrapper.getPaths(path)
        self._modelWrapper.load(paths)
        return self._modelWrapper 
    
    ## Get the best trial state from the oracle
    # @return State information of the best trial
    def get_best_trial_state(self):
        return self.oracle.get_best_trials(1)[0].get_state()
    

    ## Add some data to the log
    # @param trial_id       The id of the trial
    # @param data           The data to add to the log
    def _addToLog(self, trial_id, data):
        self._log.append({
            "trial": {
                "id": trial_id,
                "data": data
            }
        })
    
    ## Obtain the log from the algorithm
    # @return List of logged items
    def getLog(self):
        return self._log
