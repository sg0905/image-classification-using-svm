Image Classification Using Local Binary Pattern (LBP) and Uniform Local Binary Pattern (ULBP) with SVM

    This project demonstrates image classification using Local Binary Pattern (LBP) and Uniform Local Binary Pattern (ULBP) for feature extraction and Support Vector Machine (SVM) for classification. The dataset used consists of images from two categories: Ancient Egyptian architecture and Art Nouveau architecture.

Dependencies
    Ensure you have the following dependencies installed:

        Python 3.x
        NumPy
        OpenCV
        scikit-image
        scikit-learn
        matplotlib

Running the Code
    Set the path of the dataset in the data_directory variable.
    Define the categories of the dataset in the categories list.
    Run the script.

The script performs the following steps:

Reads the images in grayscale.
    Resizes the images to 50x50 pixels.
    Extracts features using Local Binary Pattern (LBP) and Uniform Local Binary Pattern (ULBP).
    Splits the dataset into training and testing sets.
    Trains a Support Vector Machine (SVM) on the training set.
    Evaluates the SVM on the testing set.
    Prints the accuracy and confusion matrix.
    Plots the confusion matrix.


This folder contains the dataset only with the Google Images (g-images) or both datasets joined (architecture-style-dataset).

You can find the dataset made by Zhe Xu on https://www.kaggle.com/wwymak/architecture-dataset.

