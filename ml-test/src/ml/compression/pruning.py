import tensorflow as tf
import tensorflow_model_optimization as tfmot


## Structured channel pruning for keras models
# @param model           A keras model
# @param ranking_method  String: 'magnitude'/'taylor' for channel ranking method
# @return                Return pruned model
def channel_pruning(model_, ranking_method='magnitude'):

    # Define the pruning parameters
    if ranking_method == 'magnitude':
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.2,
                final_sparsity=0.8,
                begin_step=0,
                end_step=1000
                ),
            'block_size': (1, 1)  # Prune entire filters (channels) at a time
            }
    elif ranking_method == 'taylor':
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.2,
                final_sparsity=0.8,
                begin_step=0,
                end_step=1000
            ),
            'n_iterations': 10,
            'mult_factor': 0.5,
            'pruning_objective': 'taylor'
        }
    else:
        raise ValueError("Invalid ranking_method parameter. Use 'magnitude' or 'taylor'.")
    
    print(pruning_params)
    print(model_.layers)
    
    # Prune the convolutional layers in the model
    for layer in model_.layers:
        print(layer)
        if isinstance(layer, tf.keras.layers.Conv1D):
            print("Pruning convolutional layer: ", layer)
            # Apply magnitude-based structured channel pruning to the layer
            if ranking_method == 'magnitude':
                pruned_layer = tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            elif ranking_method == 'taylor':
                pruned_layer = tfmot.sparsity.keras.prune_low_taylor_rank(layer, **pruning_params)
            else:
                raise ValueError("Invalid ranking_method parameter. Use 'magnitude' or 'taylor'.")
            print(pruned_layer)
            # Replace the original layer with the pruned layer in the model
            model_.get_layer(name=layer.name).set_weights(pruned_layer.get_weights())
    
    # Prune the dense layers in the model
    for layer in model_.layers:
        print(layer)
        if isinstance(layer, tf.keras.layers.Dense) and layer is not model_.layers[-1]:
            print("Pruning dense layer: ", layer)
            # Apply magnitude-based structured channel pruning to the layer
            if ranking_method == 'magnitude':
                pruned_layer = tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            elif ranking_method == 'taylor':
                pruned_layer = tfmot.sparsity.keras.prune_low_taylor_rank(layer, **pruning_params)
            else:
                raise ValueError("Invalid ranking_method parameter. Use 'magnitude' or 'taylor'.")
            print(pruned_layer)
            # Replace the original layer with the pruned layer in the model
            model_.get_layer(name=layer.name).set_weights(pruned_layer.get_weights())
    
    return model_