# Date Classification - Full Guide

This is a documentation and tutorial for creating an Azure machine learning web app which will classify an image of a date into one of the following categories:

1. PR_Class_Model 
2. PR_Skin_Model
3. PR_Waste_Model
## Table of Contents
* [Configuration](#config)
  * [Dependencies & Requirements](#d&r)
  * [Import and split your data](#import-split)
* [Training Models](#train-main)
  * [Train a custom Convolutional Neural Network](#train-custom)
  * [Train a pre-trained Convolutional Neural Network (Transfer Learning)](#train-transfer)
## <a name="config"></a>Configuration

- Download the environment that is used in this project, [Anaconda](https://www.anaconda.com/products/individual) - a great environment and dependencies manager for data science tools.
- Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest), create an Azure account and use it to create a resource group, container registry and web app.
- Install [Docker](https://www.docker.com/products/docker-desktop) (This will be used to upload our app to cloud).
- **Optional (for Nvidia GPU enabled devices)** - [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) and [CUDNN 7.6.5 for CUDA 10.1](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-101) (Do not install newer version as tensorflow supports only version 10.1 and below as of 8/23/2020)

**Note** - It is highly recommended to install all of your dependencies with Anaconda via `conda install` command to avoid conflicts. Using conda-forge channel is also recommended. To add it, type in the terminal:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
```
### <a name="d&r"></a>Dependencies & Requirements
  - python=3.7
  - tensorflow-gpu
  
    **Our models will be built using TensorFlow API, I suggest using _tensorflow-gpu_ (If you have a GPU) dependency that performs the calculations on your Nvidia GPU usage for better performance overall.**
  
  - flask
  - matplotlib
  - pillow
  - numpy
  - split-folders (optional)

### <a name="import-split"></a>Import and split your data
While not necessary, I recommend using pip library split-folders.
To install, simply type this command in terminal: `pip install split-folders`
The process is very simple and requires only two lines of code
```
import splitfolders as sf
sf.ratio(in_path, out_path, seed=1337, ratio=(.7, .2, .1))
```
This will basically take all of the subfolders from `in_path` and will split the data to a new folder that is located in `out_path` with the ratio of (train-70%, validation-20%, test-10%).
## <a name="train-main"></a>Training models
This reopository includes two unique methods of training
### <a name="train-custom"></a>Training a custom Convolutional Neural Network
To build a custom CNN, you will be using `custom_CNN.py` module in the project.

The main functions that you will have to use are: import_data(), train_model(), score().
If you have some knowledge or want to experiment with the models you can easily change parameters of your model in train_model().
I would suggest tinkering with `learning_rate` of the optimizer, `filter_size` and `batch_size` or even changing the structure of the CNN.

After you're done with setting up the model to your liking, create a folder under the project's directory with `save_path`'s value as the name.
**Note** - Remember to change values of train/val/test paths according to the folder name where you will be importing data from (`out_path` if you've used the splitfolders method )

And now it's showtime! To build and train your model type in:
```
train_gen, val_gen = import_data()
train_model(train_gen, val_gen)
```
Later on you can use it again and even classify new data using the following method:
```
model = load_model(join(home, save_path, model_name))
model.load_weights(join(save_path, weights_path))
result = score(filepath, filename)
```
This code will load your trained model and the weights that performed the best during training - It's a good habit to save and load your models, just to make sure they are not corrupted and moreover, you don't want to train a new model everytime when you will be deploying your app to the web, it will be computationally expensive and time wasting.

The model will classify the new image and will return a JSON object. Here's an example: `{"confidence":0.9818311333656311,"label":"PR_Class_Model"}`

If you want to get even deeper, and analyze your models, feel free to check the other functions that are in this module. **Note** - You can see a graph of your model's performance using [Tensorboard - Tensorflow's visualization toolkit](https://www.tensorflow.org/tensorboard/). You can run Tensorboard terminal by typing `tensorboard --logdir yourpath/to/logs`. I suggest creating a logging dir under `save_path` folder.
### <a name="train-transfer"></a>Training a pre-trained Convolutional Neural Network (Transfer Learning)
**To use a predefined model that was trained on a large dataset beforehand, you will be using `transfer_learning.py` for this task.
This module works very similarly to `custom_CNN.py`, so you can follow the same steps in the previous paragraph.****

Although the workflow is pretty much the same, you can see some interesting differences between the two methods.
Due to previously training on a large dataset, this model will get a much better headstart than the former model. In contrast, its improvement will decay faster until it reaches a point when the custom model actually performs better (My hypothesis is that the model can't improve too much since all of its pretrained parts are frozen, and the top layer is just insufficient to provide full adaptation to this problem; this is where fine-tuning can actually help). If you want to use this project for more general purposes, like classifying animals, I am sure this method might will provide better results than a custom model. Since this project has a more specific purpose, transfer learning doesn't work in our favour in this case.
## <a name="flask"></a> Using the model in a flask app
In this section we create a web app on a local machine and we begin automating the process and making a pipeline.
The code for creating the app is provided in app.py. You are not required to do much, just choose the names for your upload folder and model names.

--TO BE CONTINUED
