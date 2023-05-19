import os
import numpy as np
import json

# Set paths to predefined datasets
SNR1_path = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/Dataset/final1_n100"
SNR2_path = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/Dataset/final2_n100"
SNR3_path = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/Dataset/final3_n100"
SNR4_path = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/Dataset/final4_n100"
SNR5_path = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/Dataset/final5_n100"
# Set path to new total dataset
SNRtotal_path = "C:/Users/maris/Documents/DataScience/Thesis/PinPoach_Thesis/Data/Dataset/total_8k"

# Create list of all dataset variable names
SNR_list = [SNR1_path, SNR2_path, SNR3_path, SNR4_path, SNR5_path]

# Set paths to data and layers
data_SNR1_path = os.path.join(SNR1_path, "data.npz") 
layer_SNR1_path = os.path.join(SNR1_path, "layers.json")
data_SNR2_path = os.path.join(SNR2_path, "data.npz") 
layer_SNR2_path = os.path.join(SNR2_path, "layers.json")
data_SNR3_path = os.path.join(SNR3_path, "data.npz") 
layer_SNR3_path = os.path.join(SNR3_path, "layers.json")
data_SNR4_path = os.path.join(SNR4_path, "data.npz") 
layer_SNR4_path = os.path.join(SNR4_path, "layers.json")
data_SNR5_path = os.path.join(SNR5_path, "data.npz") 
layer_SNR5_path = os.path.join(SNR5_path, "layers.json")

# Load data
data1 = np.array(np.load(data_SNR1_path)["data"]) 
data2 = np.array(np.load(data_SNR2_path)["data"]) 
data3 = np.array(np.load(data_SNR3_path)["data"]) 
data4 = np.array(np.load(data_SNR4_path)["data"]) 
data5 = np.array(np.load(data_SNR5_path)["data"]) 

# Load layers
with open(layer_SNR1_path, "r") as jsonFile:
    layer1 = json.loads(jsonFile.read())
with open(layer_SNR2_path, "r") as jsonFile:
    layer2 = json.loads(jsonFile.read())
with open(layer_SNR3_path, "r") as jsonFile:
    layer3 = json.loads(jsonFile.read())
with open(layer_SNR4_path, "r") as jsonFile:
    layer4 = json.loads(jsonFile.read())
with open(layer_SNR5_path, "r") as jsonFile:
    layer5 = json.loads(jsonFile.read())
    
# Change data format to single list of tuples
dataset1 = list(zip(data1, layer1))
dataset2 = list(zip(data2, layer2))
dataset3 = list(zip(data3, layer3))
dataset4 = list(zip(data4, layer4))
dataset5 = list(zip(data5, layer5))

# Combine and shuffle all datasets
total_dataset = dataset1 + dataset2 + dataset3 + dataset4 + dataset5
np.random.shuffle(total_dataset)

# Create empty lists to store total data and layers
total_data = []
total_layer = []

# Split the total dataset in data and layers again
total_data = np.array([sample[0] for sample in total_dataset])
print(total_data[:3])
total_layer = [sample[1] for sample in total_dataset]
print(total_layer[:3])

# Save the data and layers tot total directory
np.savez_compressed(os.path.join(SNRtotal_path, "data.npz"), data=total_data)
with open(os.path.join(SNRtotal_path, "layers.json"), 'w+') as jsonFile:
    json.dump(total_layer, jsonFile, indent=4)
