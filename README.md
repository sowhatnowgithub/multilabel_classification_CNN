# Meesho Attribute Prediction - Multilabel Classification with CNN

Contributors : Myself and [epoch-seeker](https://github.com/Epoch-Seeker)

This repository contains code for the **Meesho Attribute Prediction Competition**, where the goal is to predict various product attributes for e-commerce items using image and metadata. The solution utilizes a custom Convolutional Neural Network (CNN) built from scratch and incorporates a basic data cleaning procedure.

The original code which we used for the compettion is in this link https://github.com/Epoch-Seeker/Meesho-Attribute-Predict-Competition/blob/main/README.md
## Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Getting Started](#getting-started)
- [Data Cleaning](#data-cleaning)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Overview

In this project, we perform **multilabel classification** to predict product attributes for different categories of e-commerce products. We use both image data (product photos) and tabular metadata (product attributes) for training. The solution involves:

1. Cleaning the data to handle missing values.
2. Encoding the categorical attributes using integer encoding.
3. Designing a custom CNN model to handle the image data and predict the attribute labels for the different categories.
4. Implementing data augmentation techniques to improve model performance.

### Original Code

The original code was written by me in collaboration with [epoch-seeker](https://github.com/Epoch-Seeker).

The project repository can be found here: [Meesho Attribute Predict Competition](https://github.com/Epoch-Seeker/Meesho-Attribute-Predict-Competition/)

## Key Innovations

- **Custom CNN Implementation**: A convolutional neural network (CNN) was written from scratch to address the specific requirements of the multilabel classification task.
- **Brute Force Data Cleaning**: A simple yet effective data cleaning approach was implemented to handle missing values. Various strategies were tested iteratively to determine the best method for the task.
- **Image Augmentation**: Random transformations, such as flipping, rotation, and color jittering, were applied to increase data diversity and improve model generalization.

## Getting Started

To run this project on your local machine, follow these steps:

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- TensorFlow (for model training)
- scikit-learn (for data preprocessing)
- matplotlib (for visualizations)
- Kaggle API (for downloading competition data)

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/Epoch-Seeker/Meesho-Attribute-Predict-Competition.git
   cd Meesho-Attribute-Predict-Competition
```

# Visual Taxonomy Attribute Prediction

## Setup Instructions

### 1. Install Required Dependencies
To install the necessary Python packages, run the following command:

```bash
pip install -r requirements.txt
```
2. Set up Kaggle API for Dataset Download

Go to your Kaggle account.

Create a new API token.

Download the kaggle.json file.

Place the kaggle.json file in the root directory of your project.

Run the following commands to configure the Kaggle API:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

3. Download and Extract the Dataset

To download the dataset from the Kaggle competition and extract it, run:

```
kaggle competitions download -c visual-taxonomy
unzip visual-taxonomy.zip
```

Data Preparation

The dataset contains images and a CSV file with metadata about product categories and attributes. The code provided reads and preprocesses the data, handling missing values and encoding categorical attributes with integer encoding.

Model Training

The model is a custom Convolutional Neural Network (CNN) implemented using TensorFlow/Keras. It is designed to predict multiple product attributes for each item using the image as input.

To train the model, you can call the train_model function with the training and validation data:

```
train_model(x_train, y_train, x_val, y_val, model, epochs=10)
```

During training, the model learns to predict multiple product attributes using the provided image data.

Image Augmentation

To improve model performance, we use augmentation techniques such as:

Random horizontal and vertical flips

Rotations

Color adjustments

These augmentations help increase the diversity of the training data.

Model Architecture

The CNN model consists of the following layers:

Convolutional Layers: To extract features from the images.

Max Pooling: To reduce the dimensionality of the feature maps.

Batch Normalization: To speed up training and improve generalization.

Dropout: To prevent overfitting.

Fully Connected Layers: For final classification of attributes.

For each attribute category, a separate output layer is defined to predict the corresponding attribute values using a softmax activation function

```
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def define_model(n_classes, input_shape, each_class_attr):
    input_layer = Input(shape=input_shape)

    # First Convolutional Block
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Second Convolutional Block
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Third Convolutional Block
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    # Output Layers for Each Attribute Category
    output_layers = []
    for i in range(n_classes):
        y = Dense(each_class_attr[i], activation='softmax', name=f'attr_{i+1}_output')(x)
        output_layers.append(y)
    
    model = Model(inputs=input_layer, outputs=output_layers)
    model.compile(optimizer=Adam(0.0002), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

Results

The model was trained and evaluated on multiple product categories. The performance was monitored using accuracy metrics for each attribute category. Data augmentation and tuning of model parameters, such as the learning rate, were crucial to achieving good results.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright and Attribution

This project contains code and ideas developed by the contributors of the Meesho Attribute Prediction competition. The original code was written by me in collaboration with epoch-seeker. The Kaggle competition dataset is hosted by Kaggle.

Note: This repository is for educational purposes and research only. The code and models should not be used for commercial purposes without prior permission from the original authors and Kaggle.



