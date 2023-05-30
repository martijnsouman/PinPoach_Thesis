# import os
# import tensorflow as tf
# from tensorflow import keras

# # https://www.tensorflow.org/lite/performance/post_training_quant

# # Model path
# path = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/ml-test/src/models/final5_n100/HyperConv1_Dense1_1DModel" 

# # Load the model
# model = keras.models.load_model(os.path.join(path, 'model'))
# print(model)

# # Convert the Keras model to a TensorFlow Lite model
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# print(converter)

# # Set the optimization flag for quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # Convert the model to TensorFlow Lite format
# tflite_model = converter.convert()

# # Load the quantized model into a new interpreter
# interpreter = tf.lite.Interpreter(model_content=tflite_model)
# interpreter.allocate_tensors()





# class CustomConv1DQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
#     def get_weights_and_quantizers(self, layer):
#         return [(layer.kernel, 
#                  tfmot.quantization.keras.quantizers.LastValueQuantizer(
#                      num_bits=8, 
#                      symmetric=True, 
#                      narrow_range=False, 
#                      per_axis=False))]

#     def get_activations_and_quantizers(self, layer):
#         return [(layer.activation, 
#                  tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
#                      num_bits=8, 
#                      symmetric=True, 
#                      narrow_range=False, 
#                      per_axis=False))]

#     def set_quantize_weights(self, layer, quantize_weights):
#         layer.kernel = quantize_weights[layer.kernel]

#     def set_quantize_activations(self, layer, quantize_activations):
#         layer.activation = quantize_activations[layer.activation]
        
#     def get_output_quantizers(self, layer):
#         # Does not quantize output, since we return an empty list.
#         return []
    
#     def get_config(self):
#         return {}


# def model_quantization(model):

#     # Use custom function for Conv1D layers and normal quantization for others
#     for layer in model.layers:
#         if isinstance(layer, tf.keras.layers.Conv1D):
#             quantize_config = CustomConv1DQuantizeConfig()
#             tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=quantize_config)
#         else:
#             tfmot.quantization.keras.quantize_annotate_layer(layer)
    
#     # Apply quantization
#     quantized_model = tfmot.quantization.keras.quantize_apply(model)

#     return quantized_model
