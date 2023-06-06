import os
import pathlib
import tensorflow as tf
from tensorflow import keras

from ml.compression.sparsification import *  # When only running this file: from sparsification import *
from ml.compression.pruning import *  # When only running this file: from pruning import *

# # Quantize the model by turning it into a Tensorflow Lite model
# # @param
# # @return
# def quantization_tflite(model, model_path):
    
#     # Convert the Keras model to a TensorFlow Lite model
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     tflite_model = converter.convert()    
    
#     # Create directory to story model
#     tflite_models_dir = pathlib.Path(os.path.join(model_path, 'quantized_model'))
#     tflite_models_dir.mkdir(exist_ok=True, parents=True)
    
#     # Store the model
#     tflite_model_file = tflite_models_dir/"tflite_model"
#     tflite_model_file.write_bytes(tflite_model)    
    
#     return 


# # https://www.tensorflow.org/lite/performance/post_training_quant

# # Load model
# modelPath = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/ml-test/src/models/total_2k/HyperConv2_Dense1_1DModel"
# model = keras.models.load_model(os.path.join(modelPath, 'model'))

# # Sparsification
# sparse_model = l0_sparse_pruning(model, 0.5)

# # Pruning
# pruned_model = channel_pruning(sparse_model)

# # Convert the Keras model to a TensorFlow Lite model
# converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
# tflite_model = converter.convert()

# # Create directory to story model
# tflite_models_dir = pathlib.Path(os.path.join(modelPath, 'quantized_model'))
# tflite_models_dir.mkdir(exist_ok=True, parents=True)

# # Store the model
# tflite_model_file = tflite_models_dir/"tflite_model"
# tflite_model_file.write_bytes(tflite_model)

# # Load the model in an interpreter
# interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
# interpreter.allocate_tensors()




