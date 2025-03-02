# Human Emotion Detection

## Overview

This project focuses on detecting human emotions from facial expressions using deep learning techniques. The FER2013 dataset is utilized for training and evaluating the model. The project aims to classify emotions such as happiness, sadness, anger, surprise, and more, making it applicable in various fields like mental health monitoring, customer experience analysis, and human-computer interaction.

![Human Emotion Detection Interface](https://drive.google.com/uc?export=view&id=1IdRTtFfrxHhIFSCEpJJNjpnj8UuAsooq
)

## Dataset

The FER2013 (Facial Expression Recognition 2013) dataset contains grayscale images of facial expressions, each labeled with one of the following emotions:

- Angry

- Disgust

- Fear

- Happy

- Sad

- Surprise

- Neutral

## Dataset Details:

Number of images: 35,887

Image size: 48x48 pixels

Split: Training (28,709 images), Validation/Test (7,178 images)

## Features of the Project

Preprocessing: Image resizing, normalization, and data augmentation to improve model performance.

Model Architecture: A Convolutional Neural Network (CNN) designed to learn and classify facial expressions effectively.

Metrics: Accuracy to evaluate model performance.

Deployment: Integration with a GUI for real-time emotion detection.

## Project Workflow

### 1. Data Preprocessing

Resizing all images to 48x48 pixels.

Normalizing pixel values to [0, 1] range.

Augmenting data using techniques like rotation, flipping, and zooming.

### 2. Model Design

CNN Layers:

Convolutional layers with ReLU activation.

MaxPooling layers to reduce spatial dimensions.

Dropout layers to prevent overfitting.

Fully Connected Layers: Dense layers for classification.

Output Layer: Softmax activation for multi-class emotion classification.

### 3. Model Training

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Epochs: 50 (adjustable)

Batch Size: 64

### 4. Evaluation

Testing the model on the validation/test set.

Generating confusion matrices and classification reports.

### 5. Deployment (Optional)

Creating a simple GUI or integrating the model into an API for real-time predictions.


## Future Enhancements

Extend the model to detect multiple emotions in group images.

Integrate with video streams for real-time emotion detection.

Improve model accuracy using transfer learning.

## Contributors

Arpita Goel
