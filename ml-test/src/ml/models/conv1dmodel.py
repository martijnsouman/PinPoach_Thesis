# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np

def Conv1DModel(xTrain, yTrain, xTest, yTest):

    model = Sequential()
    # A Sequential model is appropriate for a plain stack of layers where each 
    # layer has exactly one input tensor and one output tensor.


    # Convolutional layer
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(20000, 1)))

    # Max pooling layer
    model.add(MaxPooling1D(pool_size=2))

    # Flatten layer
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(16, activation='relu'))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=10, batch_size=8)
    
    
    # Make prediction
    yPred = model.predict(xTest)
    print(yPred)
    yPred = np.round(yPred)
    print(yPred)
    
    # Calculate accuracy
    accuracy = np.mean(yPred == yTest)
    print("Accuracy: ", accuracy)
