from ..abstractmodels import *

## AbstractConvolutionalModel class
# 
# Used to extend the functionalities of the AbstractModel class with
# specific convolutional model functions
class AbstractConvolutionalModel(AbstractModel):

    ## Visualize the convolution kernels
    def plotConvolutionKernels(self):
        self._verbosePrint("Plotting convolution kernels")
    
        for layer in self._model.layers:
            # Check for convolutional layer
            if 'conv' not in layer.name:
                continue

            # Get filter weights
            filters, biases = layer.get_weights()
            n_filters = layer.filters
            
            self._verbosePrint("Visualizing kernels(n=" + str(n_filters) + ") of layer: " + str(layer.name) + str(filters.shape))

            # Normalize filter values
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
             
            # plot first few filters
            for i in range(n_filters):
                # get the filter
                f = filters[:, :, :, i]
                
                # specify subplot and turn of axis
                ax = plt.subplot(n_filters, 1, i+1)
                ax.set_xticks([])
                ax.set_yticks([])

                # plot filter channel in grayscale
                plt.imshow(f[:, :, 0], cmap='gray')

            # show the figure
            plt.show()
