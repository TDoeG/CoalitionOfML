# GrayscaleToColorizedML
This project uses a deep learning model to colorize grayscale images. The model is implemented in PyTorch and consists of an encoder-decoder architecture designed to predict color channels from grayscale input images.

## Features
- Deep Learning Model: Implements an encoder-decoder network for image colorization.
- PyTorch Implementation: Efficient and flexible model training and evaluation using PyTorch.
- Data Preprocessing: Includes scripts for loading, processing, and preparing images for training and evaluation.

## Project Structure
```
GrayScaleToColorizedML/
│
├── model.py            # Contains the ColorizationModel class
├── training.py         # Contains the training, evaluation, and prediction code
├── data_processing.py  # Contains image loading and preprocessing functions
├── visualization.py    # Contains functions for visualizing the predictions
└── assets/
    ├── images/
    │   ├── results/    # Contains the training images (grayscale)
    │   └── images/     # Contains the images for colorization (color)
    └── path_to_image   # Example image for testing
```

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

## Data Structure
1. **Training Images**: Place grayscale images in the assets/images/results/ directory. These images are used for training.

2. **Color Images**: Place the corresponding color images in the assets/images/ directory. These images provide the color information for the training.

## How to Run the Model
1. **Prepare Data**: Ensure that your grayscale and color images are placed in the correct directories as outlined in the Data Structure section.

2. **Train the Model**: Run the training script to train the model on your dataset.

```
python training.py
```
3. **Predict and Visualize**: The script will also handle predicting the colors for a sample image and visualize the results.
Make sure to update the path_to_image in training.py to the path of an image you want to test the model on.

## Usage
After training the model, you can use it to colorize new grayscale images. Update the path_to_image variable in training.py to the path of the grayscale image you wish to colorize. The script will handle preprocessing, prediction, and visualization of the output.