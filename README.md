# Date Classification

An Azure machine learning web app which will classify an image of a date into one of the following categories:

1. PR_Class_Model
2. PR_Skin_Model
3. PR_Waste_Model

## Configuration

- Download the environment that is used in this project, [Anaconda](https://www.anaconda.com/products/individual), a great environment and dependencies manager for data science tools.
- Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest), create Azure account, create a resource group, container registry and web app.
- Install Docker (This will be used to upload our app to cloud)
- **Optional (for Nvidia GPU devices)** - [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) and [CUDNN 7.6.5 for CUDA 10.1](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-101) (Do not install newer version as tensorflow supports only version 10.1 and below as of 8/23/2020)

**Note** - It is highly recommended to install all of your dependencies with Anaconda via `conda install` command to avoid conflicts. Using conda-forge channel is also recommended, to add it type in the terminal
```
conda config --add channels conda-forge
conda config --set channel_priority strict
```
### Dependencies
  - python=3.7
  - tensorflow-gpu
  
    **Our models will be built using TensorFlow API, I suggest using _tensorflow-gpu_ (If you have a GPU) dependency that performs the calculations on your Nvidia GPU usage for better performance overall.**
  
  - flask
  - matplotlib
  - pillow
  - numpy
  - split-folders (optional)

### Import and splitting data
While not necessary, I recommend using pip library split-folders.
To install, simply type this command in terminal: `pip install split-folders`
The process is very simple and requires only two lines of code
```
import splitfolders as sf
sf.ratio(in_path, out_path, seed=1337, ratio=(.7, .2, .1))
```
This will basically take all of the subfolders from `in_path` and will split the data to a new folder that is located in `out_path` with the ratio of (train-70%, validation-20%, test-10%)
## Training models
This reopository includes two unique methods of training
## Building a custom Convolutional Neural Network
To build a custom CNN, open custom_CNN.py

--- TO BE CONTINUED
