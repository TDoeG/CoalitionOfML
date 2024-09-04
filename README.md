# CoalitionOfML
These projects uses deep learning/convolution neural networks in order to do various goals. The folders named **Original_Model** and **New_Model** are used for grayscale images to color conversion using deep learning neural networks. The folder named **THE_NEWEST_MODEL** pertains the classification of the CIFAR-10 dataset. (In the future, the CIFAR-100 dataset)

## Features
- Deep Learning Model: Implements an encoder-decoder network for image colorization.
- PyTorch Implementation: Efficient and flexible model training and evaluation using PyTorch.

## File Structure
The project file structure is broken into three main folders: **Original_Model** and **New_Model**. The Original Model is deprecated and doesn't work as the approach to the neural network is not good. The New Model is the new and working apporach to the neural network and is only trained on the CIFAR-10 dataset (32x32 images).

### > Original and New Model
The Original Model and the New Model both have folders that contains saved models/trained models, after the training process. Each model has an assets folder containing the assets (or planned assets) we want to use during our training/testing of the models.

The Original Model is less organized than the New Model, as each section is split up into different python files for easier debugging and reading.

***These folders/models are abandoned/deprecated/obsolete since these models use the CIFAR-10 dataset to predict the color space of an grayscale image. WE ARE NO LONGER WORKING ON THESE. These models are not guarenteed to work at all.***

### > The Newest Model
The Newest Model has a data folder, which contains the CIFAR-10 and the CIFAR-100 datasets. The saved models are saved in Saved_Models. There are 3 main files:

- main.py & main_backup.py : *These files contains the loading of the dataset, training the model, and saving the model into the local folder.*

- model.py : *This file contains the convolutional neural network used to classify the CIFAR-10 dataset to an specified accuracy.*

- demo.py : *This file contains the demo portion of the model, made for presentational work for a general audience.*

## Installation
Use the following command to install the dependecies (Old dependencies):

```
pip install numpy torch torchvision scikit-image opencv-python pillow matplotlib
```
- NumPy: For numerical operations and array manipulation.
- PyTorch: The core deep learning library for building and training the model.
- TorchVision: Provides utilities for image processing and loading.
- scikit-image: Used for image processing tasks such as converting RGB to LAB and grayscale.
- OpenCV: For image manipulation tasks such as resizing.
- Pillow: For opening and processing images.
- Matplotlib: For visualizing the results.

```
Python Version: 3.10
```

## How to Run the Model
1. **Prepare Data**: Ensure that your dataset is ready in the assets folder or what not.

2. **Train the Model**: Run `main.py` script to download the dataset, run the model, train it, and visualize the precitions from the provided model.

3. **Demo the Model**: Run `demo.py` script to demo the model (The demo picks a random image from the CIFAR-10 or the CIFAR-100 dataset and uses those rnadomly selected images to classify which class it is in using a locally saved model.) 

    ***ONLY FOR THE_NEWEST_MDOEL***