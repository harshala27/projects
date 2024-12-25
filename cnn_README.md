# MNIST Classification using PyTorch CNN

## Overview
This project implements a Convolutional Neural Network (CNN) in PyTorch to classify handwritten digits from the MNIST dataset. It also evaluates model performance using metrics such as precision, recall, and F1-score. Additionally, it includes functionality to test predictions on custom images loaded from Google Drive.

## Features
- CNN model for digit classification with dropout and batch normalization.
- Training and testing pipelines for the MNIST dataset.
- Evaluation metrics: Confusion matrix, precision, recall, and F1-score.
- Image prediction functionality for external images using Google Drive.

## Requirements
- Python 3.x
- PyTorch
- Torchvision
- OpenCV
- Matplotlib
- NumPy
- PIL (Pillow)
- Google Colab (Optional for testing external images)

## Dataset
- MNIST dataset is automatically downloaded and used for training and testing.
- Custom images can be loaded from Google Drive for further evaluation.

## Model Architecture
The CNN consists of:
1. **Convolutional Layers:** Extract features using 16 and 32 filters with ReLU activation.
2. **Pooling Layers:** Downsample features using MaxPooling.
3. **Fully Connected Layers:** Classify features into 10 output categories.
4. **Dropout and Batch Normalization:** Prevent overfitting and speed up training.

## Usage

### 1. Install Dependencies
```bash
pip install torch torchvision numpy matplotlib pillow opencv-python-headless
```

### 2. Train the Model
1. Download and load MNIST dataset.
2. Train the model with 5 epochs using the following command:
```bash
python cnn_mnist.py
```

### 3. Evaluate Performance
- The model prints training loss and accuracy per epoch.
- Computes and displays the test accuracy.
- Generates confusion matrix and metrics (Precision, Recall, F1-Score).

### 4. Test Custom Images
1. Upload images to Google Drive.
2. Update the image path in the script.
3. Predict classes for uploaded images.

### Example Output:
```plaintext
Epoch 1/5, Loss: 0.3456, Accuracy: 89.45%
Epoch 2/5, Loss: 0.1892, Accuracy: 94.67%
...
Test Accuracy: 96.78%
Confusion Matrix:
[[580   0   2 ...]
 [  0 612   3 ...]
 ...]
Class 0: Precision=0.98, Recall=0.97, F1-Score=0.97
Average Precision: 0.96
Average Recall: 0.96
Average F1-Score: 0.96
```

## Custom Image Classification
1. Place test images in Google Drive under a folder (e.g., `test_images`).
2. Ensure Colab is used, and Google Drive is mounted.
3. Execute the prediction script to test.

### Example Output for Custom Images:
```plaintext
Image: /path/to/image1.png, Predicted Class: 7
Image: /path/to/image2.png, Predicted Class: 3
```

## Results
- High accuracy achieved on MNIST dataset.
- Ability to classify custom images effectively.

## Future Enhancements
- Implement data augmentation for further performance improvement.
- Add more visualizations for feature maps.
- Extend the model to handle larger datasets.
