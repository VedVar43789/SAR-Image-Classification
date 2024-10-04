# SAR Image Classification Project

Table of Contents
Project Overview
Project Structure
Dataset
Preprocessing
Model Architecture
Speckle Filtering
Training the Model
Evaluation
Installation and Usage
Results
Contributing
License
Contact
Project Overview

This project involves classifying Synthetic Aperture Radar (SAR) images into four distinct land cover classes:

Agriculture
Barren
Grassland
Urban
The primary challenge lies in the noisy nature of SAR images, primarily due to speckle noise. To address this, speckle filtering techniques like Lee and Gamma MAP filters are applied, followed by Convolutional Neural Network (CNN)-based image classification. This project uses a custom-built CNN model and aims to maximize accuracy, precision, and recall while maintaining computational efficiency.



# Project Structure

The project structure is organized as follows:
Dataset

The dataset comprises 16,000 images in total, with 4,000 images per class. The data is stored in the following structure:

dataset/train/agriculture/: Contains 4000 SAR images of agricultural land.
dataset/train/barren/: Contains 4000 SAR images of barren land.
dataset/train/grassland/: Contains 4000 SAR images of grassland.
dataset/train/urban/: Contains 4000 SAR images of urban areas.


All images are preprocessed using speckle filters before being fed into the model for training.



# Preprocessing

Speckle noise is a common issue in SAR images, degrading the quality and affecting classification accuracy. The following preprocessing steps are applied to each image:

Rescaling: Each image is resized to 224x224 pixels to match the input size of the CNN model.
Speckle Filtering:
Lee Filter: Applied to reduce speckle noise by using the local statistics of each pixel's neighborhood.
Gamma MAP Filter: An adaptive filter that uses the probability distribution of the speckle to reduce noise.
Normalization: The pixel values are normalized to [0, 1].
The preprocessing steps can be found in the data_preprocessing.ipynb notebook.



# Model Architecture

The model used for this project is a custom-built Convolutional Neural Network (CNN) consisting of several layers:

Input Layer: Accepts 224x224 grayscale SAR images.
Conv2D Layers: Three convolutional layers with ReLU activation for feature extraction.
MaxPooling2D Layers: Pooling layers to reduce the dimensionality of the feature maps.
Flatten Layer: Converts the 2D feature maps into 1D vectors.
Dense Layers: Two fully connected layers to map the features to the output classes.
Output Layer: A softmax layer for classifying the input image into one of the four classes.
The model architecture is defined in src/model.py.



# Speckle Filtering

Two key filters are implemented to reduce noise in the SAR images:

Lee Filter: This filter computes the local mean and variance for noise reduction while preserving edges.
Gamma MAP Filter: A more advanced filter that uses the multiplicative noise model, particularly suited for SAR image analysis.
The filters are implemented in src/speckle_filters.py.



# Training the Model

The model is trained using the following configurations:

Loss Function: Categorical Cross-Entropy
Optimizer: Adam
Metrics: Accuracy, Precision, Recall
Batch Size: 32
Epochs: 10
Learning Rate: 0.001 (fixed)
The training is performed using TensorFlow/Keras and can be reproduced by running the model_training.ipynb notebook.



# Evaluation

The model is evaluated on various performance metrics, including:

Accuracy: Measures the overall correctness of the predictions.
Precision: The ratio of true positives to the sum of true and false positives.
Recall: The ratio of true positives to the sum of true positives and false negatives.
Confusion Matrix: To evaluate the performance across each class.
You can find the evaluation results in model_training.ipynb.



# Installation and Usage

Prerequisites
Ensure you have the following installed:

Python 3.8+
TensorFlow 2.x
Keras
OpenCV
NumPy
Matplotlib
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/VedVar43789/sar-image-classification.git
cd sar-image-classification
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Download and prepare the dataset (if not already available).
Running the Project
To preprocess the dataset and train the model:

Open and run the data_preprocessing.ipynb notebook to apply speckle filtering and prepare the dataset.
Run the model_training.ipynb notebook to train the model and evaluate its performance.
Results



# The following are the key results from the SAR image classification model:

Accuracy: 85% on the test dataset.
Precision: 0.86 (average across all classes).
Recall: 0.84 (average across all classes).
Confusion Matrix: The model performed well in agriculture and urban classification but had some misclassifications in barren and grassland categories.
For detailed results, refer to the model_training.ipynb notebook.


# Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request with detailed descriptions of your changes. Contributions related to improving accuracy, implementing more advanced speckle filters, or optimizing model performance are highly appreciated!



# Contact

For questions, issues, or further information, feel free to contact me:

Name: Vedant Vardhaan
Email: vvardhaan@gmail.com
