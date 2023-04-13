# PinPoach_Thesis
Compressing a gunshot audio detecting Convolutional Neural Network for the use on edge devices.

This project contains two main folders:
- EDA: A folder containing all notebooks and plots used for the data preprocessing.
    - EDA_MarissaFaas: A jupyter notebook containing an Exploratory Data Analysis for a thesis deadline of the UvA.
- Data: A folder containing all notebooks and files needed for the data processing:
    - dataset_gather: A jupyter notebook for downloading all dataset elements (background noise and gunshot signals).
    - gunshots, YouTube, UrbanSound8K: The raw audio .wav files in their origional lenths
    - SplitFiles: All audio files split into snippets of 10 seconds (african_savanna_day, african_savanna_night, rain, thunder, and single_shots)
    - Dataset: The final augmented datasets per test run in data.npz (compressed numpy files) with layers.json file explaining the used layers per sample.
- ml-test: A folder containing all notebooks needed for data augmentation, creating, training and validating the model