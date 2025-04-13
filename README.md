# Deep Learning Labs

This repository contains solutions to the first two deep learning lab assignments, including work with text data, image preprocessing, and CSV-based data visualization.

## Lab Summaries

### Lab 1 - One-Hot Encoding from Text Data

In this lab, the goal was to process the content of 10 books from the Gutenberg dataset, convert the text into character sequences, and generate training data for neural networks.  
Each character was transformed into a one-hot encoded vector, and the final output was saved as a 3D NumPy array for future use.

### Lab 2 - Image Augmentation & CSV Visualization

The first part of this lab focused on augmenting an image dataset to prepare it for image classification tasks by applying transformations like cropping, color shifts, and rotations.  
The second part involved preprocessing a CSV dataset by handling missing values, converting units, removing duplicates, and visualizing the data using matplotlib with plots like histograms, boxplots, and a custom chart.

## Folder Structure Example

```
DeepLearning/
├── Lab1/
│   └── PrepareDataToOneHotEncodedBool.py
├── Lab2/
│   ├── csvVisualization.py
│   └── imageAugmentation.py
└── README.md
```
