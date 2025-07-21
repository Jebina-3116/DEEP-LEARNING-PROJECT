# DEEP-LEARNING-PROJECT

COMPANY:CODTECH IT SOLUTIONS

NAME:JEBINA SELIN S.D

INTERN ID:CT04DG3456

DOMAIN:DATA SCIENCE

DURATION:4 WEEKS

MENTOR:NEELA SANTOSH

Objective:
The primary objective of this project is to build a custom image classification model using Convolutional Neural Networks (CNNs) with TensorFlow. The model is trained to distinguish between two categories: cats and dogs. The pipeline starts by downloading and organizing image data from the web, continues through data preprocessing, model training, and evaluation, and ends with real-time predictions on test images. This end-to-end approach simulates a real-world deep learning project pipeline.

Motivation:
Image classification is one of the core problems in computer vision. The goal here is to develop a complete deep learning workflow starting from data collection to prediction, mimicking real-world AI applications like pet recognition apps, animal monitoring systems, or image-based search engines.

 Tools and Technologies Used:
Python: Core programming language for scripting.

TensorFlow / Keras: Used for building, training, and evaluating the CNN model.

PIL (Pillow): For image processing and resizing.

Matplotlib: For visualizing sample images and accuracy plots.

Scikit-learn: For data splitting (train_test_split).

Requests + BytesIO: For downloading images from web URLs.

 Pipeline Steps and Explanation:
🔹 1. Data Collection (Download Images):
The dataset is created manually by downloading images from given URLs. Two classes are defined: cat and dog, each with a few sample images. The script organizes the images into a directory structure suitable for TensorFlow:

bash
Copy
Edit
my_dataset/
├── train/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
Each image is resized to 128×128 pixels and saved using Pillow.

 2. Data Visualization:
Before training, a few images from the training dataset are visualized using matplotlib to confirm that the data has been downloaded and structured correctly.

 3. Data Preprocessing with TensorFlow:
The images are loaded into TensorFlow datasets using image_dataset_from_directory() for both the training and validation sets. Batch size is set to 32, and image size to (128, 128). After loading:

Pixel values are normalized from [0, 255] to [0.0, 1.0] using a .map() function.

Class names are automatically inferred from the directory names.

 4. Model Architecture:
A Convolutional Neural Network (CNN) is constructed using tf.keras.Sequential. The architecture includes:

2 convolutional layers with ReLU activation followed by max-pooling.

A flatten layer to convert 2D features to 1D.

A dense hidden layer with 64 units and ReLU.

A final dense layer with neurons equal to the number of classes (len(class_names)).

 5. Compilation and Training:
The model is compiled using:

Optimizer: Adam

Loss Function: SparseCategoricalCrossentropy with from_logits=True

Metric: Accuracy

The model is trained for 5 epochs on the custom dataset, and both training and validation accuracy are recorded.

 6. Performance Visualization:
After training, accuracy values for both training and validation sets are plotted to visualize the model's learning progress.

 7. Prediction and Evaluation:
A batch of test images is passed to the model, and predictions are made using model.predict(). The predicted and true labels are displayed alongside the corresponding images to evaluate performance visually.

 Results and Observations:
The CNN model successfully distinguishes between cat and dog images even with a small, custom dataset.

Visualization of predictions shows that the model learns meaningful features from the images despite the limited training data.

Accuracy can be improved by increasing dataset size, adding data augmentation, and training for more epochs.

 Key Learning Outcomes:
How to create a custom dataset by downloading and labeling images.

Structuring image folders for TensorFlow compatibility.

Preprocessing images for deep learning models.

Building and training CNN models in TensorFlow.

Visualizing predictions and evaluating model performance.

 Conclusion:
This project successfully demonstrates an end-to-end image classification pipeline using a small, manually created dataset. It mimics a real-world deep learning scenario involving data collection, preprocessing, training, and evaluation. The experience gained here is highly transferable to larger classification problems and forms a strong foundation for future projects involving more advanced computer vision tasks.

OUTPUT:

