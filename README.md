# GrayscaleToColorizedML
This project uses a deep learning model to colorize grayscale images. The model is implemented in PyTorch and consists of an encoder-decoder architecture designed to predict color channels from grayscale input images.

## Features
- Deep Learning Model: Implements an encoder-decoder network for image colorization.
- PyTorch Implementation: Efficient and flexible model training and evaluation using PyTorch.
- Data Preprocessing: Includes scripts for loading, processing, and preparing images for training and evaluation.

## File Structure
The project file structure is broken into two main folders: **Original_Model** and **New_Model**. The Original Model is deprecated and doesn't work as the approach to the neural network is not good. The New Model is the new and working apporach to the neural network and is only trained on the CIFAR-10 dataset (32x32 images).

The Original Model and the New Model both have folders that contains saved models/trained models, after the training process. Each model has an assets folder containing the assets (or planned assets) we want to use during our training/testing of the models.

The Original Model is less organized than the New Model, as each section is split up into different python files for easier debugging and reading.

## Installation
Use the following command to install the dependecies:

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
1. **Prepare Data**: Ensure that your dataset

2. **Preprocessing Data**: Run `data_preprocessing.py` script to preprocess your images into cleaner, much more ingestible data.

```
python data_preprocessing.py
```


3. **Train the Model**: Run `main.py` script to train the model on your dataset.

```
python training.py
```

4. **Predict and Visualize**: The script will also handle predicting the colors for a sample image and visualize the results.
Make sure to update the path_to_image in training.py to the path of an image you want to test the model on.

## Usage
After training the model, you can use it to colorize new grayscale images. Update the path_to_image variable in training.py to the path of the grayscale image you wish to colorize. The script will handle preprocessing, prediction, and visualization of the output.