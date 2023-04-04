import tensorflow as tf
import numpy as np
import random

def setUseGPU(value):
    if value:
        #Set memory growth to true for all GPU devices
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        #Disable GPU usage
        tf.config.set_visible_devices([], "GPU")

def setSeed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
