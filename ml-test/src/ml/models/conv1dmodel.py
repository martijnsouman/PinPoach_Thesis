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
from keras.models import Sequential
from keras.optimizers import SGD

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
        num_epochs=5):
    
    # Build all different combinations of models
    for num_conv_layers in conv_layer_range:
        for num_dense_layers in dense_layer_range:
            
            # Create model name
            model_name = 'Conv{}_Dense{}_1DModel'.format(
                num_conv_layers, num_dense_layers)
            
            # Create model path
            individual_output_path = os.path.join(output_path, model_name)
            
            # Create validation set 
            # train : val : test = 0.55 : 0.2 : 0.25 
            val_size = int(0.2666 * len(x_train))
            x_val = x_train[0:val_size]
            y_val = y_train[0:val_size]
            layer_val = layer_train[0:val_size]
            x_train = x_train[val_size:]
            y_train = y_train[val_size:]
            layer_train = layer_train[val_size:]
            
            # Build model
            Conv1DModel = buildConv1DModel(
                num_conv_layers, 
                num_dense_layers,
                x_train)
            
            start_train_time = time.time()
            
            # Train model
            history = Conv1DModel.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=8)
            # Save training time
            train_duration = time.time()-start_train_time
            
            start_pred_time = time.time()
            
            # Model prediction
            y_pred = Conv1DModel.predict(x_test)
            # Save prediction time
            pred_duration = time.time()-start_pred_time
            y_pred = np.round(y_pred)
            
            # Evaluate model
            acc, pres, rec, f1_s, cm_df = evaluateConv1DModel(Conv1DModel, 
                                                              x_test, 
                                                              y_test, 
                                                              y_pred, 
                                                              labels)
               
            # Save model
            storeConv1DModel(individual_output_path, 
                             Conv1DModel, 
                             model_name, 
                             acc,
                             pres,
                             rec,
                             f1_s,
                             cm_df,
                             train_duration,
                             pred_duration)
    
    
    return


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
                     presicion, 
                     f1_sc, 
                     cm_df, 
                     training_time,
                     prediction_time):
    # Try to create the directory
    try:
        os.mkdir(path)
    except:
        pass
    
    # # Write model to json file
    # model_json = model.to_json() 
    # with open(os.path.join(path, 'model'), "w") as jsonFile:
    #     jsonFile.write(model_json)
        
    # # Write weights to a json file
    # model.save_weights(os.path.join(path, 'weights'))
    
    # Save the model
    model.save(os.path.join(path, 'model'))
    
    # Create dataframe with accuracy and training time
    df = pd.DataFrame(columns=['Parameter', 'Value'])
    df = df.append({'Parameter': 'model_name', 'Value': model_name}, ignore_index=True)
    df = df.append({'Parameter': 'accuracy', 'Value': accuracy}, ignore_index=True)
    df = df.append({'Parameter': 'recall', 'Value': recall}, ignore_index=True)
    df = df.append({'Parameter': 'presicion', 'Value': presicion}, ignore_index=True)
    df = df.append({'Parameter': 'f1_score', 'Value': f1_sc}, ignore_index=True)
    df = df.append({'Parameter': 'training_time', 'Value': training_time}, ignore_index=True)
    df = df.append({'Parameter': 'prediction_time', 'Value': prediction_time}, ignore_index=True)
    
    # Write performance dataframe to csv
    df.to_csv(os.path.join(path, 'performance.csv'), index=False)
    
    # Write confustion matrix to png 
    sn.heatmap(cm_df, annot=True, annot_kws={"size": 32})
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(path, 'confusion_matrix'))

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

    # Add input layer
    model.add(kl.Input(shape=np.shape(xTrain)[1:]))
    
    #Add convolution layers
    for i in range(convLayer):
        # possibly specify number of filerters
        # possibly specify the activation function
        
        convLayer = kl.Conv1D(
            filters=32, # could make this HyperParameter for HyperModel
            kernel_size=3, # could make this HyperParameter for HyperModel
            activation='relu',  # could make this HyperParameter for HyperModel
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
        numUnits = 250  # could make this HyperParameter for HyperModel
        activationFunction = 'relu' # could make this HyperParameter for HyperModel

        model.add(kl.Dense(units=numUnits, activation=activationFunction, kernel_initializer='he_uniform'))
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
    
def buildSimpleConv1DModel(convLayer, denseLayer, xTrain): 
    
    keras.backend.clear_session()
    
    # Define input shape
    print(np.shape(xTrain)[1:])
    
    # THIS WORKS
    # # Define model
    # model = Sequential([
    #     kl.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
    #     kl.MaxPooling1D(pool_size=2),
    #     kl.Flatten(),
    #     kl.Dense(units=1, activation='sigmoid')
    #     ])
    
    # THIS WORKS
    model = Sequential()

    # Add input layer
    model.add(kl.Input(shape=np.shape(xTrain)[1:]))
    # Add conv layers
    convLayer = kl.Conv1D(filters=32, kernel_size=3, activation='relu', data_format='channels_last', padding='same')
    model.add(convLayer)
    model.add(kl.MaxPooling1D(pool_size=2))
    model.add(kl.Dropout(0.5)) 
    model.add(kl.BatchNormalization()) 
    model.add(kl.Flatten())
    
    # Add dense layer  
    numUnits = 250  # could make this HyperParameter for HyperModel
    activationFunction = 'relu' # could make this HyperParameter for HyperModel
    model.add(kl.Dense(units=numUnits, activation=activationFunction, kernel_initializer='he_uniform')) 
    model.add(kl.Dropout(0.5))
    
    # Add dense output layers
    model.add(kl.Dense(units=1, activation='sigmoid'))

    # Compile model
    # THIS WORKS
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  

    # Print model summary
    model.summary()
    
    return model



