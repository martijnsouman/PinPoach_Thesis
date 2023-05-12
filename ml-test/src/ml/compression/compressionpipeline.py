import os
from tensorflow import keras
import numpy as np

from ml.compression.sparsification import *


def mainCompression(models_path,
                    x_train,
                    y_train,
                    x_test, 
                    y_test, 
                    layer_train, 
                    layer_test):
    
    # Load all models
    for rootPath, dirNames, files in os.walk(models_path):
        print(rootPath)
        print(dirNames)
        print(files)
        for modelName in dirNames:
            print(modelName)
            print(os.path.join(rootPath, modelName, 'model'))
            model = keras.models.load_model(os.path.join(rootPath, modelName, 'model'))
            print(model)
            
            model_performance(model, x_test, y_test)
            
            # Sparsify
            # Perform pruning on the model's weights
            sparse_model = l0_sparse_pruning(model, 0.5)
            
            model_performance(sparse_model, x_test, y_test)
            
            # Prune
            # - Magnitude based ranking
            #-  Taylor criteria based ranking
            
            
            # Quantization


## Check model perfomance 
# @param model          Keras model
# @param x_test         Test set
# @return               Print y_pred and accuracy
def model_performance(model, x_test, y_test):
    
    # Prediction
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)
    print("Prediction: ", y_pred)
    
    # Accruacy
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, batch_size=8)
    print(accuracy)


