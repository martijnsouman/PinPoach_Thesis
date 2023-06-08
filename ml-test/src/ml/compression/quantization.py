import os
import pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np

# # When running entire code: 
# from ml.compression.sparsification import * 
# from ml.compression.pruning import *   
# When only running this file: 
from sparsification import *
from pruning import *


# When only running this file: 


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


# https://www.tensorflow.org/lite/performance/post_training_quant

# Load model
modelPath = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/ml-test/src/models/total_2k/HyperConv2_Dense1_1DModel/new_sparse_model"
# pruned_model = keras.models.load_model(os.path.join(modelPath, 'model'))

# Convert the Keras model to a TensorFlow Lite model
# converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
# tflite_model = converter.convert()

# Create directory to story model
tflite_models_dir = pathlib.Path(os.path.join(modelPath, 'quantized_model'))
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Store the model
tflite_model_file = tflite_models_dir/"tflite_model"
# tflite_model_file.write_bytes(tflite_model)

# Alternative to above 4 lines
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir/"tflinte_quant_model_.tflite"
# tflite_model_quant_file.write_bytes(tflite_quant_model)


# Make prediction
# Load the model in an interpreter
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()  # Needed before execution

interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()  # Neede before execution

# load x_test and y_test
np_save_dir = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data"
x_test = np.load(np_save_dir + "/xTest.npy") # (400, 20000, 1)
y_test = np.load(np_save_dir + "/yTest.npy") # (400, )

# Initialize variables for evaluation
num_samples = x_test.shape[0] # 400 
y_true = np.squeeze(y_test)  # Remove unnecessary dimensions -> stays same

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluateTfliteModel(interpreter):
  # Get in- and output dimensions
  input_shape = interpreter.get_input_details()[0]["index"] # 0
  output_shape = interpreter.get_output_details()[0]["index"] # 39

  # Initialize lists to store evaluation metrics and prediction times
  accuracy_list = []
  precision_list = []
  recall_list = []
  f1_score_list = []
  prediction_time_list = []
  y_pred = []
  
  # Run predictions on every image in the "test" dataset.
  for sample in x_test:
    # Pre-processing: reshape sample to (1, 20000, 1)
    input_data = np.expand_dims(sample, axis=0)  
    input_data = input_data.astype(np.float32)
    interpreter.set_tensor(input_shape, input_data) 
    
    # Run inference.
    interpreter.invoke()

    # Post-processing: extract prediction from interpreter
    output = interpreter.tensor(output_shape)
    prediction = int(np.round(output()[0]))
    y_pred.append(prediction) 

  # Calculate accuracy 1
  accurate_count = 0
  for index in range(len(y_pred)):
    if y_pred[index] == y_test[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(y_pred)
  print(accuracy)
  
  # Calculate accuracy 2
  accurate_count = 0
  accurate_count = np.sum(np.array(y_pred) == y_test)
  print(accurate_count)
  accuracy = accurate_count / len(y_pred)
  print(accuracy)

  return accuracy

print(evaluateTfliteModel(interpreter))
print(evaluateTfliteModel(interpreter_quant))

# CHAT GPT
# # Iterate over the test set
# for i in range(num_samples):        
#     # Get input shape
#     input_shape = input_details[0]['shape']
    
#     # Preprocess input data
#     input_data = x_test[i].reshape(input_shape).astype(np.float32)
    
#     # Run inference on the original model
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     start_time = time.time()
#     interpreter.invoke()
#     end_time = time.time()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
    
#     # Run inference on the quantized model
#     interpreter_quant.set_tensor(input_details_quant[0]['index'], input_data)
#     start_time_quant = time.time()
#     interpreter_quant.invoke()
#     end_time_quant = time.time()
#     output_data_quant = interpreter_quant.get_tensor(output_details_quant[0]['index'])
    
#     # Postprocess output data
#     y_pred = np.squeeze(output_data)
#     y_pred_quant = np.squeeze(output_data_quant)
    
#     # Calculate evaluation metrics
#     accuracy = np.mean(y_pred == y_true[i])
#     precision = sklearn.metrics.precision_score(y_true[i], y_pred)
#     recall = sklearn.metrics.recall_score(y_true[i], y_pred)
#     f1_score = sklearn.metrics.f1_score(y_true[i], y_pred)
    
#     # Calculate prediction time
#     prediction_time = end_time - start_time
#     prediction_time_quant = end_time_quant - start_time_quant
    
#     # Store evaluation metrics and prediction times
#     accuracy_list.append(accuracy)
#     precision_list.append(precision)
#     recall_list.append(recall)
#     f1_score_list.append(f1_score)
#     prediction_time_list.append(prediction_time)

# # Calculate average evaluation metrics and prediction time
# average_accuracy = np.mean(accuracy_list)
# average_precision = np.mean(precision_list)
# average_recall = np.mean(recall_list)
# average_f1_score = np.mean(f1_score_list)
# average_prediction_time = np.mean(prediction_time_list)

# # Print and save the evaluation metrics and prediction time
# print("Accuracy: ", average_accuracy)
# print("Precision: ", average_precision)
# print("Recall: ", average_recall)
# print("F1-score: ", average_f1_score)
# print("Average Prediction Time: ", average_prediction_time)




