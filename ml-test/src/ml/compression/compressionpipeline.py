import os
from tensorflow import keras
import tensorflow_model_optimization as tfmot
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
            
            # Try out new sparsity
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            
            # Define end step
            batch_size = 8
            validation_split = 0.2666
            num_samples = x_train.shape[0]
            end_step = np.ceil(num_samples / batch_size).astype(np.int32) * epoch_num
            print(end_step)
            
            # Define model for pruning
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.50,
                    final_sparsity=0.90,
                    begin_step=0,
                    end_step=end_step)}
            model_for_pruning = prune_low_magnitude(model, **pruning_params)
            
            # Recompile
            optimizer = model.optimizer 
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 
                         tf.keras.metrics.Precision(), 
                         tf.keras.metrics.Recall()])
            
            # Train and evaluate
            callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
            
            start_train_time = time.time()
            model.fit(x_train, 
                                  y_train,
                                  batch_size=batch_size, 
                                  epochs=epoch_num, 
                                  validation_split=validation_split,
                                  callbacks=callbacks)
            # Save training time
            train_duration = time.time()-start_train_time
            
            # Make prediction with pruned model
            # Start prediction time
            start_pred_time = time.time()
            
            # Make prediction
            y_pred = model.predict(x_test)
            y_pred = np.round(y_pred)
            
            # Stop prediction time
            pred_duration = time.time()-start_pred_time
            
            # Evaluate pruned model
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
            
            # Prepare pruned model for saving
            pruned_model = tfmot.sparsity.keras.strip_pruning(model)
            
  
            # Save model to check size and performance
            storeConv1DModel(os.path.join(modelPath, 'new_sparse_model'), 
                             pruned_model, 
                             'new_sparse_model', 
                             accuracy, 
                             recall, 
                             precision, 
                             f1_sc, 
                             cm_df, 
                             train_duration,  
                             pred_duration)
            
            
            # # Sparsify
            # # Perform pruning on the model's weights
            # sparse_model = l0_sparse_pruning(model, 0.9)
            
            # # Evaluate and store sparse model
            # model_predict_store(sparse_model, 
            #                     'sparse_model',
            #                     modelPath,
            #                     x_test, 
            #                     y_test, 
            #                     labels)
                        
            # # Prune
            # # - Magnitude based ranking
            # #-  Taylor criteria based ranking
            # pruned_model = channel_pruning(sparse_model, channel_ranking)
            
            # # Evaluate and store pruned model
            # model_predict_store(pruned_model, 
            #                     'pruned_model',
            #                     modelPath,
            #                     x_test, 
            #                     y_test, 
            #                     labels)
            
            # # Retrain 
            # # Retrieve the optimizer used in the initial model
            # optimizer = pruned_model.optimizer  
            # # Compile the pruned model 
            # pruned_model.compile(optimizer=optimizer, 
            #                      loss='binary_crossentropy', 
            #                      metrics=['accuracy', 
            #                               tf.keras.metrics.Precision(), 
            #                               tf.keras.metrics.Recall()])
            # # Retrain the pruned model
            # start_train_time = time.time()
            # history = pruned_model.fit(x_train, y_train, # Save training time
            #                            batch_size=8,
            #                            epochs=epoch_num,
            #                            validation_split=0.2666)
            # # Save training time
            # train_duration = time.time()-start_train_time
            
            # # Try to create the directory
            # try:
            #     os.mkdir(os.path.join(modelPath, 'retrained_model'))
            # except:
            #     pass
            # # Create and store the learning curve 
            # plot_learning_curve(history, os.path.join(modelPath, 'retrained_model'))
            # # Evaluate and store retrained model
            # model_predict_store(pruned_model, 
            #                     'retrained_model',
            #                     modelPath,
            #                     x_test, 
            #                     y_test, 
            #                     labels,
            #                     train_duration)
            
            
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
    plt.show()
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, 'learning_curve_loss'))
    plt.show()
    plt.clf()

    
    
## Make a prediction, store all performance metrics and save model
# @param model          Keras model
# @param ModelName      String of model name
# @param Xtest          Test set with features 
# @param Ytest          Test set with true values
# @param labels         Dictionary of label names
# @return               Makes prediction, calls evaluateConv1DModel for 
#                       performances, storeConv1DModel to store performances
def model_predict_store(model, 
                        ModelName, 
                        model_path, 
                        Xtest, 
                        Ytest, 
                        labels, 
                        train_time=0): 
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
                     train_time,  
                     pred_duration)
    
    return 
