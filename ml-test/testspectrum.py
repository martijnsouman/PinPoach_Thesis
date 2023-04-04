import librosa
import librosa.display

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import os

from pydub import AudioSegment
from pydub.playback import play

#datasetPath = "../dataset/urbansound8k/"
datasetPath = "../dataset/test/"

count = 0

#Loop through all files in the datasetPath
for f in os.listdir(datasetPath):
    filePath = datasetPath + str(f)

    y, sr = librosa.load(filePath)
    S = np.abs(librosa.stft(y))
    print("shape: " + str(np.shape(S)))

    s_db = librosa.amplitude_to_db(S, ref=np.max)

    sx = ndimage.sobel(S, axis=0, mode='constant')
    sy = ndimage.sobel(S, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    
    
    plt.subplot(1, 2, 1)
    #fig, ax = plt.subplots()
    #img = librosa.display.specshow(s_db, x_axis='time', y_axis='log', ax=ax)
    img = librosa.display.specshow(s_db, x_axis='time', y_axis='linear')
    #ax.set(title=f)
    #fig.colorbar(img, ax=ax, format="%+2.f dB")


    plt.subplot(1, 2, 2)
    plt.imshow(sob, cmap=plt.cm.gray)
    plt.title(f)

    #play(AudioSegment.from_wav(filePath))
    plt.show()

    #count += 1
    #if(count > 5):
    #    exit()
