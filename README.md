# MNIST Classification using PyTorch MLP

## Overview
This project implements an Enhanced Multi-Layer Perceptron (MLP) in PyTorch to classify handwritten digits from the MNIST dataset. It also evaluates model performance using metrics such as precision, recall, and F1-score. Additionally, it includes functionality to test predictions on custom images loaded from Google Drive.

## Features
- Enhanced MLP model for digit classification with dropout and batch normalization.
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
The Enhanced MLP consists of:
1. **Fully Connected Layers:** Five layers with 1024, 512, 256, 128, and 10 units respectively.
2. **Activation Functions:** ReLU for non-linear transformations.
3. **Dropout Layers:** Prevent overfitting by randomly deactivating neurons.
4. **Batch Normalization Layers:** Normalize activations to improve stability and convergence.

## Usage

### 1. Install Dependencies
```bash
pip install torch torchvision numpy matplotlib pillow opencv-python-headless
```

### 2. Train the Model
1. Download and load MNIST dataset.
2. Train the model with 20 epochs using the following command:
```bash
python enhanced_mlp_mnist.py
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
Epoch 1/20, Loss: 0.4321, Accuracy: 85.67%
Epoch 2/20, Loss: 0.3124, Accuracy: 89.78%
...
Test Accuracy: 94.56%
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


