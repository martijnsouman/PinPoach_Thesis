import os
from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import quantize_model
import numpy as np

from ml.models import *
from ml.compression.sparsification import *
from ml.compression.pruning import *
from ml.compression.quantization import *

def mainCompression(models_path, 
                    x_train, 
                    y_train, 
                    x_test, 
                    y_test, 
                    layer_train, 
                    layer_test, 
                    labels, 
                    channel_ranking = 'magnitude'):
    print(x_test)
    print(x_test.shape)
    print(type(x_test))
    
    
    # Load all models
    for rootPath, dirNames, files in os.walk(models_path):
        print(rootPath)
        print(dirNames)
        print(files)
        for modelName in dirNames:  #
            print("Compressing: ", modelName)
            # Skip the other files which are always in this directory
            if modelName == 'optimized_output' or modelName == 'plots':
                print("Skipping: ", modelName)
                continue
            
            # Load model
            modelPath = os.path.join(rootPath, modelName)
            model = keras.models.load_model(os.path.join(modelPath, 'model'))
            
            # Sparsify
            # Perform pruning on the model's weights
            sparse_model = l0_sparse_pruning(model, 0.5)
            
            # Evaluate and store sparse model
            model_predict_store(sparse_model, 
                                'sparse_model',
                                modelPath,
                                x_test, 
                                y_test, 
                                labels)
                        
            # Prune
            # - Magnitude based ranking
            #-  Taylor criteria based ranking
            pruned_model = channel_pruning(sparse_model, channel_ranking)
            
            # Evaluate and store pruned model
            model_predict_store(pruned_model, 
                                'pruned_model',
                                modelPath,
                                x_test, 
                                y_test, 
                                labels)
            
            # Quantization
            quantized_model = model_quantization(pruned_model)
            
            # Prepare the input data
            input_data = x_test.astype(np.float32)

            # Get input and output details from the interpreter
            input_details = quantized_model.get_input_details()
            output_details = quantized_model.get_output_details()

            # Set the input tensor value
            quantized_model.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            quantized_model.invoke()

            # Get the output tensor value
            output_data = quantized_model.get_tensor(output_details[0]['index'])
            print(output_data)
            
            # Evaluate and store quantized model
            model_predict_store(quantized_model, 
                                'quantized_model',
                                modelPath,
                                x_test, 
                                y_test, 
                                labels)


    
    
## Make a prediction, store all performance metrics and save model
# @param model          Keras model
# @param ModelName      String of model name
# @param Xtest          Test set with features 
# @param Ytest          Test set with true values
# @param labels         Dictionary of label names
# @return               Makes prediction, calls evaluateConv1DModel for 
#                       performances, storeConv1DModel to store performances
def model_predict_store(model, ModelName, model_path, Xtest, Ytest, labels):
    # Start prediction time
    start_pred_time = time.time()
    
    # Make prediction
    Ypred = model.predict(Xtest)
    Ypred = np.round(Ypred)
    
    # Stop prediction time
    pred_duration = time.time()-start_pred_time
    
    # Evaluate model
    accuracy, precision, recall, f1_sc, cm_df = evaluateConv1DModel(
        model, 
        Xtest, 
        Ytest, 
        Ypred, 
        labels)
    
    print("Accuracy: ", accuracy)
    print("Percision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1_sc)
    
    # Save model to check size and performance
    storeConv1DModel(os.path.join(model_path, ModelName), 
                     model, 
                     ModelName, 
                     accuracy, 
                     recall, 
                     precision, 
                     f1_sc, 
                     cm_df, 
                     0,  # model has no training time
                     pred_duration)
    
    return 
