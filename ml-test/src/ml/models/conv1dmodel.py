# -*- coding: utf-8 -*-
from data import *
from ml.abstractmodels import * 

import time
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pandas as pd

from tensorflow import keras

import keras.layers as kl
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import SGD
from keras_tuner import HyperParameters
from keras_tuner.tuners import RandomSearch

from sklearn.metrics import confusion_matrix, f1_score


## Main function initiating structure and calling train and validation
# @param conv_layer_range           Range with # of convolutional layers
# @param dense_layer_range          Range with # of dense layers
# @param x_train                    Input train set
# @param y_train                    Output train set
# @param x_test                     Input test set
# @param y_test                     Output test set
# @return
def MainConv1DModel(
        output_path,
        conv_layer_range, 
        dense_layer_range,
        x_train, 
        y_train, 
        x_test, 
        y_test,
        layer_train,
        labels,
        num_epochs=5,
        hyp_model=False):
    
    # Check data balance
    counts = np.bincount(y_train)
    print("Train number of 0: ", counts[0], "number of 1: ", counts[1])
    
    # Build all different combinations of models
    for num_conv_layers in conv_layer_range:
        for num_dense_layers in dense_layer_range:
            
            # Specify what model to use
            if hyp_model == False:
                # Create model name
                model_name = 'Conv{}_Dense{}_1DModel'.format(
                    num_conv_layers, num_dense_layers)
                
                # Create model path
                individual_output_path = os.path.join(output_path, model_name)
                
                # Build model
                Conv1DModel = buildConv1DModel(
                    num_conv_layers, 
                    num_dense_layers,
                    x_train)
                
                # Train model
                start_train_time = time.time()
                #history = Conv1DModel.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=8)
                history = Conv1DModel.fit(x=x_train, y=y_train, validation_split=0.2666, epochs=num_epochs, batch_size=8)
                # Save training time
                train_duration = time.time()-start_train_time
            else:
                # For hypermodel
                model_name = 'HyperConv{}_Dense{}_1DModel'.format(
                    num_conv_layers, num_dense_layers)
            
                # Create model path
                individual_output_path = os.path.join(output_path, model_name)

                # Create tuner object
                tuner = RandomSearch(
                    lambda hp: buildConv1DHyperModel(hp, 
                                                     num_conv_layers, 
                                                     num_dense_layers, 
                                                     x_train),
                    objective='val_accuracy',
                    max_trials=10,
                    executions_per_trial=1,
                    directory=individual_output_path,
                    project_name=model_name)
                
                # Search best model
                start_train_time = time.time()
                # Only save final model
                checkpoint_path = os.path.join(individual_output_path, 'best_model.h5')
                checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
                tuner.search(x_train, 
                             y_train, 
                             validation_split=0.2666, 
                             epochs=num_epochs, 
                             batch_size=8,
                             callbacks=[checkpoint_callback])
                # Save training time
                train_duration = time.time()-start_train_time
                
                # Get best model
                Conv1DModel = tuner.get_best_models(num_models=1)[0]
                Conv1DModel_hp = tuner.get_best_hyperparameters(num_trials=1)[0] 
                print("Final Hyperparameters:")
                print(Conv1DModel_hp.values)
            
            # Model prediction
            start_pred_time = time.time()
            y_pred = Conv1DModel.predict(x_test)
            y_pred = np.round(y_pred)
            # Save prediction time
            pred_duration = time.time()-start_pred_time
            
            # Evaluate model
            acc, prec, rec, f1_s, cm_df = evaluateConv1DModel(Conv1DModel, 
                                                              x_test, 
                                                              y_test, 
                                                              y_pred, 
                                                              labels)
               
            # Save model
            storeConv1DModel(individual_output_path, 
                             Conv1DModel, 
                             model_name, 
                             acc,
                             prec,
                             rec,
                             f1_s,
                             cm_df,
                             train_duration,
                             pred_duration)
            
            # Save the hyperparameters to a file
            store_hp(Conv1DModel_hp, individual_output_path)

    return


## Try to store the hyperparameters
# @param
# @return
def store_hp(hyperparameters, directory):
    try:
        # Save the hyperparameters to a file
        hyperparameters_file = os.path.join(directory, "hyperparameters.txt")
        with open(hyperparameters_file, "w") as f:
            for param, value in hyperparameters.items():
                f.write(f"{param}: {value}\n")
    except Exception as e:
        print(f"Error occurred while storing hyperparameters: {str(e)}")


## Save the model paramters and details in files
# @param path               Path to model
# @param model              Keras model
# @param model_name         String with model name
# @param accuracy           Float of model accuracy
# @param conf_matrix        Confusion matrix as DataFrame
# @param training_time      Float of training duration
# @return                   'model.json', 'weights'.json', 
#                           and 'performance.csv' in directory 'model_name'
def storeConv1DModel(path, 
                     model, 
                     model_name, 
                     accuracy, 
                     recall, 
                     precision, 
                     f1_sc, 
                     cm_df, 
                     training_time,
                     prediction_time):
    # Try to create the directory
    try:
        os.mkdir(path)
    except:
        pass
    
    # Save the model
    model.save(os.path.join(path, 'model'))
    
    # Create dataframe with accuracy and training time
    df = pd.DataFrame(columns=['Parameter', 'Value'])
    df = df.append({'Parameter': 'model_name', 'Value': model_name}, ignore_index=True)
    df = df.append({'Parameter': 'accuracy', 'Value': accuracy}, ignore_index=True)
    df = df.append({'Parameter': 'recall', 'Value': recall}, ignore_index=True)
    df = df.append({'Parameter': 'precision', 'Value': precision}, ignore_index=True)
    df = df.append({'Parameter': 'f1_score', 'Value': f1_sc}, ignore_index=True)
    df = df.append({'Parameter': 'training_time', 'Value': training_time}, ignore_index=True)
    df = df.append({'Parameter': 'prediction_time', 'Value': prediction_time}, ignore_index=True)
    
    # Write performance dataframe to csv
    df.to_csv(os.path.join(path, 'performance.csv'), index=False)
    
    # Write confustion matrix to png 
    sn.heatmap(cm_df, annot=True, annot_kws={"size": 32}, fmt='d')
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(path, 'confusion_matrix'))
    plt.clf()

    return




## Plot and save confusion matrix for 1D model
# @param
# @return
def plotConfusionMatrix1D(matrix, labels):
    # Create dataframe
    dataframe = pd.DataFrame(matrix, index=labels, columns=labels)

    # Add heatmap
    sn.heatmap(dataframe, annot=True, annot_kws={"size": 32})
        
    # Show plot
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    plt.clf()
    
    return dataframe




## Evaluate model
# @param model              Keras model
# @param xTest              Input test set
# @param yTest              Output test set
# @param yPred              Predicted output by the model
# @param labels             Dictionary with output labels
# @return
def evaluateConv1DModel(model, xTest, yTest, yPred, labels):
    
    # Model performance
    loss, accuracy, precision, recall = model.evaluate(xTest, yTest, batch_size=8)
    f1_sc = f1_score(yTest, yPred, average='weighted')
    
    # Create confusion matrix
    confusionMatrix = confusion_matrix(yTest, yPred)
    # Plot the confusion matrix
    cm_df = plotConfusionMatrix1D(confusionMatrix, labels)
    
    return accuracy, precision, recall, f1_sc, cm_df



    
## Build the actual model
# @param convLayer                  Number of convolutional layers
# @param dense_layer_range          Number of dense layers
# @param x_train                    Input train set
# @param y_train                    Output train set
# @return
def buildConv1DModel(convLayer, denseLayer, xTrain):
    
    #Clear the session to use less RAM
    keras.backend.clear_session()
    
    # Build the model
    model = Sequential()
    # A Sequential model is appropriate for a plain stack of layers where each 
    # layer has exactly one input tensor and one output tensor.
    
    # Define hyperparameters
    hp = HyperParameters()

    # Add input layer
    model.add(kl.Input(shape=np.shape(xTrain)[1:]))
    
    #Add convolution layers
    for i in range(convLayer):
        # Define range of choise hyperparameters
        filters = hp.Choice('filters', values=[8, 16, 32, 64])
        kernel_size = hp.Choice('kernel_size', values=[2, 100])
        activation_function = hp.Choice('activation_function', values=['relu', 'sigmoid', 'tanh'])
        
        convLayer = kl.Conv1D(
            filters=filters, # could make this HyperParameter for HyperModel
            kernel_size=kernel_size, # could make this HyperParameter for HyperModel
            activation=activation_function,  # could make this HyperParameter for HyperModel
            data_format='channels_last',
            padding='same')
        
        #Add the standard layers after convolutional layer
        model.add(convLayer)
        model.add(kl.MaxPooling1D(pool_size=2))  # could make this HyperParameter for HyperModel, but is always 2
        model.add(kl.Dropout(0.5))
        model.add(kl.BatchNormalization())
    
    #Flatten the convolution data
    model.add(kl.Flatten())
    
    #Add Dense layers
    for i in range(denseLayer):
        # Define range of choise hyperparameters
        numUnits = hp.Choice('num_units', values=[50, 100, 150, 200, 250, 300, 350, 400])
        dense_activation_function = hp.Choice('dense_activation_function', values=['relu', 'sigmoid', 'tanh'])

        model.add(kl.Dense(units=numUnits, activation=dense_activation_function, kernel_initializer='he_uniform'))
        model.add(kl.Dropout(0.5))


    #Add output layer
    model.add(kl.Dense(units=1, activation='sigmoid'))
    
    #Compile model
    # # Specify range of learning rate with exponential decay
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #         initial_learning_rate=0.1,
    #         decay_steps=10000,
    #         decay_rate=0.96,
    #         staircase=True)
    # # Optimize with Stochastic Gradient Descent 
    # model.compile(
    #     optimizer=SGD(learning_rate=lr_schedule), 
    #     loss='binary_crossentropy', 
    #     metrics=['accuracy'])    
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    
    # Print architecture
    model.summary()
    
    return model

## Try to use a Hypermodel with tunable parameters
# @param
# @return   
def buildConv1DHyperModel(hp, convLayer, denseLayer, xTrain):
    
    # Build the model
    model = Sequential()
    
    # Add input layer
    model.add(kl.Input(shape=np.shape(xTrain)[1:]))
    
    # Add convolution layers
    for i in range(convLayer):
        # Define hyperparameters
        filters = hp.Choice('filters_'+str(i), values=[8, 16, 32, 64])
        kernel_size = hp.Choice('kernel_size_'+str(i), values=[2, 100])
        activation_function = hp.Choice('activation_function_'+str(i), values=['relu', 'sigmoid', 'tanh'])
        
        # Define convolutional layer
        convLayer = kl.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation_function,
            data_format='channels_last',
            padding='same')
        
        # Add layers
        model.add(convLayer)
        model.add(kl.MaxPooling1D(pool_size=2))
        model.add(kl.Dropout(0.5))
        model.add(kl.BatchNormalization())
    
    # Flatten the convolution data
    model.add(kl.Flatten())
    
    # Add dense layers
    for i in range(denseLayer):
        # Define hyperparameters
        numUnits = hp.Choice('num_units_'+str(i), values=[50, 100, 150, 200, 250, 300, 350, 400])
        dense_activation_function = hp.Choice('dense_activation_function_'+str(i), values=['relu', 'sigmoid', 'tanh'])
        
        # Add layers
        model.add(kl.Dense(units=numUnits, activation=dense_activation_function, kernel_initializer='he_uniform'))
        model.add(kl.Dropout(0.5))

    # Add output layer
    model.add(kl.Dense(units=1, activation='sigmoid'))
    
    # Define hyperparameters
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    if optimizer == 'adam':
        optim = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        optim = keras.optimizers.SGD(learning_rate=lr)
    else:
        optim = keras.optimizers.RMSprop(learning_rate=lr)
        
    # Compile model
    model.compile(optimizer=optim, loss='binary_crossentropy', 
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    
    # Print architecture
    model.summary()
    
    return model


# https://github.com/tensorflow/model-optimization/issues/362

# def buildConv2DHyperModel(hp, convLayer, denseLayer, xTrain):
    
#     # Build the model
#     model = keras.Sequential()
    
#     # Add input layer
#     model.add(kl.Input(shape=np.shape(xTrain)[1:]))
    
#     # Add convolution layers
#     for i in range(convLayer):
#         # Define hyperparameters
#         filters = hp.Choice('filters_'+str(i), values=[8, 16, 32, 64])
#         kernel_size = hp.Choice('kernel_size_'+str(i), values=[2, 100])
#         activation_function = hp.Choice('activation_function_'+str(i), values=['relu', 'sigmoid', 'tanh'])
        
#         # Add Reshape layer before Conv2D
#         model.add(kl.Reshape((np.shape(xTrain)[1], 1, 1)))
        
#         # Define convolutional layer
#         convLayer2D = kl.Conv2D(
#             filters=filters,
#             kernel_size=(kernel_size, 1),
#             activation=activation_function,
#             padding='same')
        
#         # Add convolutional layer
#         model.add(convLayer2D)
        
#         # Add Reshape layer before MaxPooling2D
#         model.add(kl.Reshape((np.shape(xTrain)[1] // 2, filters, 1)))
        
#         # Add layers
#         model.add(kl.MaxPooling2D(pool_size=(2, 1)))
#         model.add(kl.Dropout(0.5))
#         model.add(kl.BatchNormalization())
    
#     # Flatten the convolution data
#     model.add(kl.Reshape((np.shape(xTrain)[1] // 2, filters, convLayer)))
#     model.add(kl.Reshape((np.shape(xTrain)[1] // 2, filters * convLayer)))
#     model.add(kl.Flatten())
    
#     # Add dense layers
#     for i in range(denseLayer):
#         # Define hyperparameters
#         numUnits = hp.Choice('num_units_'+str(i), values=[50, 100, 150, 200, 250, 300, 350, 400])
#         dense_activation_function = hp.Choice('dense_activation_function_'+str(i), values=['relu', 'sigmoid', 'tanh'])
        
#         # Add layers
#         model.add(kl.Dense(units=numUnits, activation=dense_activation_function, kernel_initializer='he_uniform'))
#         model.add(kl.Dropout(0.5))

#     # Add output layer
#     model.add(kl.Dense(units=1, activation='sigmoid'))
    
#     # Define hyperparameters
#     optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
#     lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#     if optimizer == 'adam':
#         optim = keras.optimizers.Adam(learning_rate=lr)
#     elif optimizer == 'sgd':
#         optim = keras.optimizers.SGD(learning_rate=lr)
#     else:
#         optim = keras.optimizers.RMSprop(learning_rate=lr)
        
#     # Compile model
#     model.compile(optimizer=optim, loss='binary_crossentropy', 
#                   metrics=['accuracy', 
#                            tf.keras.metrics.Precision(),
#                            tf.keras.metrics.Recall()])
    
#     # Print architecture
#     model.summary()
    
#     return model





