import os
import pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# # When running entire code: 
# from ml.compression.sparsification import * 
# from ml.compression.pruning import *   
# When only running this file: 
#from sparsification import *
#from pruning import *


# https://www.tensorflow.org/lite/performance/post_training_quant

basePath = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/ml-test/src/models"
experimentName = "total_2k"
modelsPath = os.path.join(basePath, experimentName)
print(modelsPath)

# Load test set
np_save_dir = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data"
x_test = np.load(np_save_dir + "/xTest.npy") # (400, 20000, 1)
y_test = np.load(np_save_dir + "/yTest.npy") # (400, )

# A helper function to evaluate the TF Lite model using test dataset.
# @param
# @return
def evaluateTfliteModel(interpreter):
  # Get in- and output dimensions
  input_shape = interpreter.get_input_details()[0]["index"] # 0
  output_shape = interpreter.get_output_details()[0]["index"] # 39

  # Initialize lists to store predictions
  y_pred = []
  total_duration = 0
  
  # Run predictions on every image in the "test" dataset.
  for sample in x_test:
    # Pre-processing: reshape sample to (1, 20000, 1)
    input_data = np.expand_dims(sample, axis=0)  
    input_data = input_data.astype(np.float32)
    interpreter.set_tensor(input_shape, input_data) 
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    # Save prediction times (in seconds)
    prediction_duration = end_time - start_time
    total_duration += prediction_duration

    # Post-processing: extract prediction from interpreter
    output = interpreter.tensor(output_shape)
    prediction = int(np.round(output()[0]))
    y_pred.append(prediction) 
  
  # Evaluate tflite model
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)   
  avarage_duration = total_duration / len(y_test)
  confusionMatrix = confusion_matrix(y_test, y_pred)

  return accuracy, precision, recall, f1, avarage_duration, total_duration, confusionMatrix


# Turn a keras hypermodel into two tensorflow lite models.
# @param
# @return
def quantizeModel(model, path):
    # Convert the Keras model to a TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Create directory to story model
    tflite_models_dir = pathlib.Path(os.path.join(path, 'quantized_model'))
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Store the model
    tflite_model_file = tflite_models_dir/"tflite_model"
    tflite_model_file.write_bytes(tflite_model)
    
    # Alternative to above 4 lines with quantization on storage
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    tflite_model_quant_file = tflite_models_dir/"tflinte_quant_model_.tflite"
    tflite_model_quant_file.write_bytes(tflite_quant_model)
    
    return tflite_model_file, tflite_model_quant_file

# Store model performance
# @param
# @return
def storeTfLiteModel(path,
                     model_name,
                     accuracy, 
                     recall, 
                     precision, 
                     f1, 
                     cm_df, 
                     av_prediction_time,
                     tot_prediction_time):

    
    # Create dataframe with accuracy and training time
    df = pd.DataFrame(columns=['Parameter', 'Value'])
    df = df.append({'Parameter': 'model_name', 'Value': model_name}, ignore_index=True)
    df = df.append({'Parameter': 'accuracy', 'Value': accuracy}, ignore_index=True)
    df = df.append({'Parameter': 'recall', 'Value': recall}, ignore_index=True)
    df = df.append({'Parameter': 'precision', 'Value': precision}, ignore_index=True)
    df = df.append({'Parameter': 'f1_score', 'Value': f1}, ignore_index=True)
    df = df.append({'Parameter': 'avarage_prediction_time', 'Value': av_prediction_time}, ignore_index=True)
    df = df.append({'Parameter': 'total_prediction_time', 'Value': tot_prediction_time}, ignore_index=True)
    
    # Write performance dataframe to csv
    file_name = f'performance_{model_name}.csv'
    print(file_name)
    df.to_csv(os.path.join(path, file_name), index=False)
    
    # Write confustion matrix to png 
    sn.heatmap(cm_df, annot=True, annot_kws={"size": 32}, fmt='d')
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    file_name = f'confusion_matrix_{model_name}'
    plt.savefig(os.path.join(path, file_name))
    plt.clf()

    return

# Walk through all models
for rootPath, dirNames, files in os.walk(modelsPath):
    for modelName in dirNames:  #
        print("Quantizing: ", modelName)
        # Skip the other files which are always in this directory DELETE HERE THE OTHER MODEL NAMES
        if modelName == 'optimized_output' or modelName == 'plots':
            print("Skipping: ", modelName)
            continue

        # Load model
        modelPath = os.path.join(rootPath, modelName)
        pruned_model = keras.models.load_model(os.path.join(modelPath, 'pruned_model\model'))
        pruned_model = 1
        
        tflitePath, tflite_quantPath = quantizeModel(pruned_model, modelPath)
        
        # Load the model in an interpreter
        interpreter = tf.lite.Interpreter(model_path=str(tflitePath))
        interpreter.allocate_tensors()  # Needed before execution
        
        interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_quantPath))
        interpreter_quant.allocate_tensors()  # Neede before execution
        
        # Evaluate model
        acc, prec, rec, f1_sc, av_dur, tot_dur, cm = evaluateTfliteModel(interpreter)
        acc_quant, prec_quant, rec_quant, f1_sc_quant, av_dur_quant, tot_dur_quant, cm_quant = evaluateTfliteModel(interpreter_quant)
        
        print("Interpreter Metrics:")
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1-score:", f1_sc)
        print("Avarage prediction duration: ", av_dur)
        print("Total prediction duration: ", tot_dur)

        print("\n Quantized interpreter Metrics:")
        print("Accuracy:", acc_quant)
        print("Precision:", prec_quant)
        print("Recall:", rec_quant)
        print("F1-score:", f1_sc_quant)
        print("Avarage prediction duration: ", av_dur_quant)
        print("Total prediction duration: ", tot_dur_quant)
        
        storeTfLiteModel(os.path.join(modelPath, 'quantized_model'),
                         'tflite',
                         acc,
                         rec,
                         prec,
                         f1_sc,
                         cm,
                         av_dur,
                         tot_dur)
        
        storeTfLiteModel(os.path.join(modelPath, 'quantized_model'),
                         'tflite_quant',
                         acc_quant,
                         rec_quant,
                         prec_quant,
                         f1_sc_quant,
                         cm_quant,
                         av_dur_quant,
                         tot_dur_quant)
        

# # Load model
# modelPath = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/ml-test/src/models/total_2k/HyperConv2_Dense1_1DModel/new_sparse_model"
# pruned_model = keras.models.load_model(os.path.join(modelPath, 'model'))

# # Convert the Keras model to a TensorFlow Lite model
# converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
# tflite_model = converter.convert()

# # Create directory to story model
# tflite_models_dir = pathlib.Path(os.path.join(modelPath, 'quantized_model'))
# tflite_models_dir.mkdir(exist_ok=True, parents=True)

# # Store the model
# tflite_model_file = tflite_models_dir/"tflite_model"
# tflite_model_file.write_bytes(tflite_model)

# # Alternative to above 4 lines
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quant_model = converter.convert()
# tflite_model_quant_file = tflite_models_dir/"tflinte_quant_model_.tflite"
# tflite_model_quant_file.write_bytes(tflite_quant_model)



# # Load the model in an interpreter
# interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
# interpreter.allocate_tensors()  # Needed before execution

# interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
# interpreter_quant.allocate_tensors()  # Neede before execution

# # Load test set
# np_save_dir = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data"
# x_test = np.load(np_save_dir + "/xTest.npy") # (400, 20000, 1)
# y_test = np.load(np_save_dir + "/yTest.npy") # (400, )

# # Initialize variables for evaluation
# num_samples = x_test.shape[0] # 400 
# y_true = np.squeeze(y_test)  # Remove unnecessary dimensions -> stays same



# acc, prec, rec, f1_sc, av_dur, tot_dur, cm = evaluateTfliteModel(interpreter)
# acc_quant, prec_quant, rec_quant, f1_sc_quant, av_dur_quant, tot_dur_quant, cm_quant = evaluateTfliteModel(interpreter_quant)

# print("Interpreter Metrics:")
# print("Accuracy:", acc)
# print("Precision:", prec)
# print("Recall:", rec)
# print("F1-score:", f1_sc)
# print("Avarage prediction duration: ", av_dur)
# print("Total prediction duration: ", tot_dur)

# print("\n Quantized interpreter Metrics:")
# print("Accuracy:", acc_quant)
# print("Precision:", prec_quant)
# print("Recall:", rec_quant)
# print("F1-score:", f1_sc_quant)
# print("Avarage prediction duration: ", av_dur_quant)
# print("Total prediction duration: ", tot_dur_quant)




