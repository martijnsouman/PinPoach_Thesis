

# FIRST CODE FROM CHATGPT

import tensorflow as tf
import tensorflow_model_optimization as tfmot


def channel_pruning(model):

    # Define the pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.2,
            final_sparsity=0.8,
            begin_step=0,
            end_step=1000
        ),
        'block_size': (1, 1)  # Prune entire filters (channels) at a time
    }
    
    # Prune the convolutional layers in the model
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            # Apply magnitude-based structured channel pruning to the layer
            pruned_layer = tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            # Replace the original layer with the pruned layer in the model
            sparse_model.get_layer(name=layer.name).set_weights(pruned_layer.get_weights())
    
    # Prune the dense layers in the model
    for layer in sparse_model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer is not sparse_model.layers[-1]:
            # Apply magnitude-based structured channel pruning to the layer
            pruned_layer = tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            # Replace the original layer with the pruned layer in the model
            sparse_model.get_layer(name=layer.name).set_weights(pruned_layer.get_weights())