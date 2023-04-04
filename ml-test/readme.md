# Machine learning test

This folder contains the source code and testing project for the machine learning algorithm using the dataset downloaded by dataset-dl.

## Dependencies

The project is written in Python language. So you need to have `python3.6+` on your system and `pip3` to install the python packages.

Run the following command to install the python dependencies:
`pip3 install -r ../requirements.txt`

To run TensorFlow on a GPU, install:
- `cudnn`

## Doxyfile

The documentation of the `./src` can be generated using doxywizard or any similar tool. 

In case of using Ubuntu, follow this [link](https://www.how2shout.com/linux/how-to-install-doxygen-on-ubuntu-20-04-lts-forcal-fossa/) for installation.

Run doxywizar and specify the working directory to `<local_path>/Chengetta-Model/ml-test`. Under the tab *Run*, click *Run doxygen*, and when the generating is finished, click on *show HTML output*.


## Issues

### libcusolver.so.10

When running into issues where ``libcusolver.so.10`` cannot be found on linux, go to the path 
``/opt/cuda/lib64/`` and execute ``sudo ln -s libcusolver.so.11 libcusolver.so.10`` to fix the issue.
Also add ``export LD_LIBRARY_PATH="/opt/cuda/lib64"`` to the .bashrc  
