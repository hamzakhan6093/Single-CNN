# Single-CNN



# Convolutional Neural Network (CNN) for Image Recognition

### Project Overview
This project demonstrates the implementation of a **Convolutional Neural Network (CNN)** to recognize and classify objects in images. Inspired by Yann LeCun's pioneering work on CNNs, this project aims to provide an in-depth understanding of how CNNs process visual data through layers like convolution, ReLU, pooling, and fully connected layers.

### Key Features
1. **Image Preprocessing**:
   - Converts images into arrays of pixel values.
   - Normalizes data for better performance.

2. **Convolutional Layers**:
   - Extracts features from images using convolution operations with filters.
   - Processes multiple convolutions to identify patterns like edges, shapes, and textures.

3. **Activation Layers (ReLU)**:
   - Applies the ReLU activation function to introduce nonlinearity.
   - Converts negative values in feature maps to zero, enhancing feature detection.

4. **Pooling Layers**:
   - Downsamples feature maps to reduce computational complexity.
   - Identifies important features like edges, corners, and specific object parts.

5. **Flattening and Fully Connected Layers**:
   - Flattens the pooled feature maps into a single vector.
   - Passes the vector through fully connected layers to classify the image.

6. **Output Layer**:
   - Uses activation functions like Softmax or Sigmoid to predict probabilities for each class.
   - Outputs the final class label for the input image.

### Use Case
The CNN is implemented to classify images of Cat and Dogs, specifically identifying whether an input image is a bird or another object. 

### How It Works
1. The input image is fed into the network as a matrix of pixel values.
2. The convolutional layers process the image to extract feature maps.
3. ReLU layers rectify these feature maps to enhance nonlinearity.
4. Pooling layers downsample the data, focusing on prominent features.
5. The data is flattened and passed to fully connected layers for classification.
6. The final output is a class label with probabilities.

### Results and Performance
The CNN model demonstrates high accuracy in image classification tasks, achieving reliable performance on test datasets. Visual representations of feature maps and pooling layers are provided for a better understanding of how the model processes images.

 How to Use
1. Clone the repository and navigate to the project directory.
2. Run the script to train the CNN model on your dataset.
3. Test the model on new images to observe predictions and results.

Technologies Used
- Python
- TensorFlow/Keras (for building the CNN)
- NumPy  (visualization)

 Applications
This project showcases the use of CNNs in:
- Facial recognition on social media platforms.
- Object detection for self-driving cars.
- Medical image analysis for disease detection.

