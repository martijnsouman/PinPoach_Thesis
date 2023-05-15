import os
from tensorflow import keras
import numpy as np

from ml.compression.sparsification import *
from ml.compression.pruning import *

def mainCompression(models_path,
                    x_train,
                    y_train,
                    x_test, 
                    y_test, 
                    layer_train, 
                    layer_test,
                    channel_ranking = 'magnitude'):
    
    # Load all models
    for rootPath, dirNames, files in os.walk(models_path):
        print(rootPath)
        print(dirNames)
        print(files)
        for modelName in dirNames:
            print("Compressing: ", modelName)
            modelPath = os.path.join(rootPath, modelName)
            model = keras.models.load_model(os.path.join(modelPath, 'model'))

            #model_performance(model, x_test, y_test)
            
            # Sparsify
            # Perform pruning on the model's weights
            sparse_model = l0_sparse_pruning(model, 0.5)
            
            #model_performance(sparse_model, x_test, y_test)
            # Save the model to check size
            #sparse_model.save(os.path.join(modelPath, 'sparse_model'))
            
            # Prune
            # - Magnitude based ranking
            #-  Taylor criteria based ranking
            pruned_model = channel_pruning(sparse_model, channel_ranking)
            
            model_performance(sparse_model, x_test, y_test)
            # Save the model to check size
            pruned_model.save(os.path.join(modelPath, 'pruned_model'))
            
            
            # Quantization


## Check model perfomance 
# @param model          Keras model
# @param x_test         Test set with features
# @param y_test         Test set with true values
# @return               Print y_pred and accuracy
def model_performance(model, x_test, y_test):
    
    # Prediction
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)
    
    # Accruacy
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, batch_size=8)
    print("Accuracy: ", accuracy)

    
