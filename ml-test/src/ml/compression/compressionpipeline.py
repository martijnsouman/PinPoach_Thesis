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
                    epoch_num,
                    channel_ranking = 'magnitude'):
    
    
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
            
            # Make prediction
            y_pred = model.predict(x_test)
            y_pred = np.round(y_pred)
            
            # Evaluate model
            accuracy, precision, recall, f1_sc, cm_df = evaluateConv1DModel(
                model, 
                x_test, 
                y_test, 
                y_pred, 
                labels)
            
            print("Accuracy: ", accuracy)
            print("Percision: ", precision)
            print("Recall: ", recall)
            print("F1-score: ", f1_sc)
            
            # Sparsify
            # Perform pruning on the model's weights
            sparse_model = l0_sparse_pruning(model, 0.2)
            
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
            
            # Retrain 
            # Retrieve the optimizer used in the initial model
            optimizer = pruned_model.optimizer  
            # Compile the pruned model 
            pruned_model.compile(optimizer=optimizer, 
                                 loss='binary_crossentropy', 
                                 metrics=['accuracy', 
                                          tf.keras.metrics.Precision(), 
                                          tf.keras.metrics.Recall()])
            # Retrain the pruned model
            history = pruned_model.fit(x_train, y_train, 
                                       batch_size=8,
                                       epochs=epoch_num,
                                       validation_split=0.2666)
            
            # Make prediction
            y_pred = pruned_model.predict(x_test)
            y_pred = np.round(y_pred)
            
            # Evaluate the retrained model on the test data
            # Create new directory called retrained_model
            # save the learning curve and other normal stats to this. 
            plot_learning_curve(history, os.path.join(modelPath, 'pruned_model'))
            test_loss, test_accuracy, test_precision, test_recall = pruned_model.evaluate(x_test, y_test)
            print("Test Loss:", test_loss)
            print("Test Accuracy:", test_accuracy)
            print("Test Precision:", test_precision)
            print("Test Recall:", test_recall)
            
            # # Quantization
            # quantization_tflite(pruned_model, modelPath)
            
            # # Evaluate and store quantized model
            # # model_predict_store(quantized_model, 
            # #                     'quantized_model',
            # #                     modelPath,
            # #                     x_test, 
            # #                     y_test, 
            # #                     labels)


## Plot learning curve to investigate training
# @param                     Training history
# @return                    Two learning curve plots of accuracy and loss
def plot_learning_curve(history, path):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, 'learning_curve_accuracy'))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, 'learning_curve_loss'))

    
    
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
