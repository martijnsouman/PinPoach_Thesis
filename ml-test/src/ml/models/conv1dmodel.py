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

from sklearn.metrics import confusion_matrix



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
        labels):
    
    print(conv_layer_range)
    print(dense_layer_range)
    print(output_path)
    
    # Build all different combinations of models
    for num_conv_layers in conv_layer_range:
        print(num_conv_layers)
        for num_dense_layers in dense_layer_range:
            print(num_dense_layers)
            
            # Create model name
            model_name = 'Conv{}_Dense{}_1DModel'.format(
                num_conv_layers, num_dense_layers)
            print(model_name)
            # Create model path
            individual_output_path = os.path.join(output_path, model_name)
            print(individual_output_path)
            
            # ? Create validation set so train:val:test = 0.55:0.2:0.25 ?
            # Option 1 with function from data.datasethandler
            x_val, y_val, x_train, y_train, layer_val, layer_train = dataHandler.splitInputAndOutputLists(x_train, y_train, layer_train, 0.2666)
            ## Option 2 in this notebook
            #val_size = int(0.2666 * len(x_train))
            #x_val = x_train[0:val_size]
            #y_val = y_train[0:val_size]
            #layer_val = layer_train[0:val_size]
            #x_train = x_train[val_size:]
            #y_train = y[val_size:]
            #layer_train = layer_train[val_size:]
            
            print("x_val.shape: ", x_val.shape)
            print("y_val.shape: ", y_val.shape)
            print("x_train.shape: ", x_train.shape)
            print("y_train.shape: ", y_train.shape)
            
            # Build model
            Conv1DModel = buildConv1DModel(
                num_conv_layers, 
                num_dense_layers,
                x_train)
            
            start_time = time.time()
            
            # Train model
            Conv1DModel.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=8)
            # Save training time
            train_duration = time.time()-start_time
        
            # Model prediction
            y_pred = Conv1DModel.predict(x_test)
            print(y_pred)
            y_pred = np.round(y_pred)
            print(y_pred)
            
            # Evaluate model
            evaluateConv1DModel(Conv1DModel, x_test, y_test, y_pred, labels)
               
            # Save model
            storeConv1DModel(individual_output_path, Conv1DModel)
    
    
    return


## Save the model paramters and details in files
# @param path               Path to model
# @param model              Keras model
# @param model_name         String with model name
# @param accuracy           Float of model accuracy
# @param training_time      Float of training duration
# @return                   'model.json', 'weights'.json', 
#                           and 'performance.csv' in directory 'model_name'
def storeConv1DModel(path, model, model_name, accuracy, training_time):
    # Try to create the directory
    try:
        os.mkdir(path)
    except:
        pass
    return
    
    # Write model to json file
    model_json = model.to_json() 
    with open(path["model"], "w+") as jsonFile:
        jsonFile.write(model_json)
        
    # Write weights to a json file
    model.save_weights(path["weights"])
    
    # Create dataframe with accuracy and training time
    df = pd.DataFrame(columns=['Parameter', 'Value'])
    df = df.append({'Parameter': 'model_name', 'Value': model_name}, ignore_index=True)
    df = df.append({'Parameter': 'accuracy', 'Value': accuracy}, ignore_index=True)
    df = df.append({'Parameter': 'training_time', 'Value': training_time}, ignore_index=True)
    
    # Write performance dataframe to csv
    df.to_csv(path["performance"], index=False)

    return


## Plot and save confusion matrix for 1D model
# @param
# @return
def plotConfusionMatrix1D(matrix, labels, path):
    #Create dataframe
    dataframe = pd.DataFrame(matrix, index=labels, columns=labels)
    print(dataframe)

    #Add heatmap
    sn.heatmap(dataframe, annot=True, annot_kws={"size": 32})
        
    #Show plot
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(path)
    plt.show()
    plt.clf()
    

## Evaluate model
# @param model              Keras model
# @param xTest              Input test set
# @param yTest              Output test set
# @param yPred              Predicted output by the model
# @param labels             Dictionary with output labels
# @return
def evaluateConv1DModel(model, xTest, yTest, yPred, labels):
    
    # Model accuracy
    loss, accuracy = model.evaluate(xTest, yTest, batch_size=8)
    print("Accuracy: ", accuracy)
    
    # Create confusion matrix
    #confusionMatrix = tf.math.confusion_matrix(
    #    labels=yTest, 
    #    predictions=yPred).numpy()
    confusionMatrix = confusion_matrix(yTest, yPred)
    print(confusionMatrix)
    
    # Plot the confusion matrix
    plotConfusionMatrix1D(confusionMatrix, labels)
    
    return

    
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
        
        #Add the standard layers
        model.add(convLayer)
        model.add(kl.MaxPooling1D(2))  # could make this HyperParameter for HyperModel, but is always 2
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
    model.add(kl.Dense(1, activation='sigmoid'))
    
    #Compile model
    # Specify range of learning rate with exponential decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.1,
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True)
    # Optimize with Stochastic Gradient Descent 
    model.compile(
        optimizer=SGD(learning_rate=lr_schedule), 
        loss='binary_crossentropy', 
        metrics=['accuracy'])    
    
    # Print architecture
    model.summary()
    
    return model
    
    

