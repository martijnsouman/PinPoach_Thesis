import tensorflow as tf

from keras.models import Model
from tensorflow.keras.layers import Layer, Dense
from keras import backend as K
from keras.utils import get_custom_objects
from keras.constraints import Constraint


## L0-Norm based sparse pruning to make matrix sparse
# @param model          Keras model to prune
# @percentage           Fraction of the weights to prune
# @return               Sparse Keras model
def l0_sparse_pruning(model_, percentage):
    
    print(model_.layers)
    for layer in model_.layers:
        if isinstance(layer, Layer):
            if not layer is model_.layers[-1]:
                print("Sparsifying layer: ", layer)
                for weight in layer.trainable_weights:
                    weight.assign(Sparse(percentage)(weight))
           
    return model_


## Sparse class, Constrains the weights to be sparse
# @param constraint                   
# @return w              Weight matrix w pruned 
class Sparse(Constraint):

    def __init__(self, percentage):
        self.percentage = percentage

    def __call__(self, w):
        # Calculate total number of weights in matrix w
        num_params = K.cast(K.prod(K.shape(w)), 'float32')  
        # Calculate number of weights to be pruned with percentage
        num_prune = K.cast(num_params * self.percentage, 'int32')  
        # Absolute values of weight matrix w
        w_abs = K.abs(w)  
        # Compute threshold value from what to prune
        threshold = tf.math.top_k(tf.reshape(w_abs, [-1]), k=num_prune, sorted=True)[0][-1]  
        # Create binary mask that is true for weights above threshold
        mask = K.cast(w_abs >= threshold, K.floatx())  
        # Use mask to set all other weights to 0
        w.assign(w * mask) 
        
        return w




