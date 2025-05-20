# Deep Learning Labs

This repository contains solutions to the first two deep learning lab assignments, including work with text data, image preprocessing, and CSV-based data visualization.

## Lab Summaries

### Lab 1 - One-Hot Encoding from Text Data

In this lab, the goal was to process the content of 10 books from the Gutenberg dataset, convert the text into character sequences, and generate training data for neural networks.  
Each character was transformed into a one-hot encoded vector, and the final output was saved as a 3D NumPy array for future use.

### Lab 2 - Image Augmentation & CSV Visualization

The first part of this lab focused on augmenting an image dataset to prepare it for image classification tasks by applying transformations like cropping, color shifts, and rotations.  
The second part involved preprocessing a CSV dataset by handling missing values, converting units, removing duplicates, and visualizing the data using matplotlib with plots like histograms, boxplots, and a custom chart.

### Lab 3 - Function & Dataset Regression

Training a model to regress a custom-designed mathematical function and applying linear regression on a real Kaggle dataset. Explored issues like exploding/vanishing gradients and model limitations.

### Lab 4 - EMNIST Letter Classification

Built a simple fully connected neural network to classify EMNIST letters using only linear layers, with training monitored via TensorBoard.

### Lab 5 - CNN for Custom Dataset

Adapted a convolutional neural network to classify a custom image dataset, with support for image resizing and data augmentation to improve accuracy.

### Lab 6 - CIFAR-10 Custom CNN

Designed a custom CNN from scratch for CIFAR-10 image classification, with performance tracking in TensorBoard and accuracy-based scoring criteria.

### Lab 7 - PyTorch ResNet Transfer Learning 

Adapted a pretrained PyTorch ResNet50 neural network for Brain Tumor MRI dataset classification.

Final accuracy on the test set: 99.16%  
Accuracy for glioma class: 99.00%  
Accuracy for meningioma class: 97.71%  
Accuracy for notumor class: 99.75%  
Accuracy for pituitary class: 100.00%

## Folder Structure Example

```
DeepLearning/
├── Lab1/
│   └── PrepareDataToOneHotEncodedBool.py
├── Lab2/
│   ├── csvVisualization.py
│   └── imageAugmentation.py
├── Lab3/
│   ├── customFunctionRegression.py
│   └── datasetLinearRegression.py
├── Lab4/
│   └── emnistLetterClassification.py
├── Lab5/
│   ├── customCNNClassficationModels.py
│   └── customCNNClassficationTrain.py
├── Lab6/
│   ├── cifar10CustomCNNModels.py
│   └── cifar10CustomCNNTrain.py
├── Lab7/
│   └── transferLearning.py
└── README.md
```
