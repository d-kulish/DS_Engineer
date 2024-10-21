# DS Engineer 
This repository contains three independent data science projects, each addressing a distinct task:
1. Island Counting: Counts the number of connected components in a binary image; 
2. Tabular Regression: Predicts a continuous target variable based on tabular features; 
3. MNIST Classifier: Classifies handwritten digits from the MNIST dataset; 

Prerequisites:
1. Python: 3.11.9
2. Virtual Environment: Use a virtual environment to isolate project dependencies.
3. Libraries: Refer to requirements.txt for a complete list of required libraries.
4. Project Structure: Each task is housed in its own folder for better organization and modularity.

# Island Counting
Project Location: islands folder

Description: This Jupyter Notebook implements a function to count the number of connected landmasses (islands) within a 2D grid. Land is represented by 1, and water by 0. A depth-first search algorithm is employed to explore and mark visited landmasses.

# Regression on Tabular Data

Project Location: regression folder

Description: This project utilizes a Jupyter Notebook to build a regression model for predicting a continuous target variable from a dataset containing five groups of features. Due to time constraints, no Principal Component Analysis (PCA) or feature engineering was performed.

Key Steps:
1. Exploratory Data Analysis (EDA): The training data (train.csv) comprises features with mixed scales across five groups.
2. Feature Scaling: All features were scaled using a MinMaxScaler. This scaler is saved locally for future use.
3. Model Selection: 
    - Three algorithms were evaluated: Linear Regression, XGBoost, and DNN (Keras).
    - Training results achieved: 
    | Model                     | MSE       | R-Squared | 
    |---------------------------|-----------|-----------| 
    | Keras                     | 0.08      | 1.00%     | 
    | XGBoost                   | 0.17      | 1.00%     | 
    | Linear Regression         | 841.89    | -0.00%    |
4. Model Training (train.py): 
    - Loads training data and saved scaler.
    - Preprocesses data.
    - Trains and saves a DNN Keras model.
    - Saves training metrics in a JSON file.
5. Prediction (predict.py):
    - Loads data for prediction, saved scaler, and trained model.
    - Preprocesses data.
    - Makes predictions and saves results to results/final_preds.csv.

# Digital Classifier
Project Location: digital_classifier folder (or similar)

Description: This project builds and utilizes three models for classifying handwritten digits from the MNIST dataset: Random Forest, Convolutional Neural Network (CNN), and a Random Class Selector (baseline). Additionally, it offers a user-interactive Python application for choosing an algorithm and making predictions.

Key Components:
1. Model Training (class.ipynb):
    - Implements three independent classes for each model type: Random Forest, CNN, and Random Selector.
    - Each class performs the following:
        a. Downloads the Keras MNIST dataset.
        b. Preprocesses data for training.
        c. Trains and saves the corresponding model.
        d. Demonstrates prediction on sample images from a locally stored MNIST gallery.
2. Classification Library (classifier):
    - Provides methods for all three selected algorithms.
    - Method functionality:
        a. Accepts a raw image as input.
        b. Preprocesses the image for the chosen model.
        c. Loads the relevant saved model.
        d. Makes a classification prediction.
        e. Prints the image and the predicted class.
3. User Application (app.py):
    - A simple Python application that interacts with the user.
    - Functionalities:
        a. Asks the user to choose a classification algorithm (Random Forest, CNN, or Random Selector).
        b. Randomly selects an image from a local folder.
        c. Makes a class prediction for the selected image using the chosen algorithm.
        d. Prints the predicted class.

Important Note: Pre-trained models and sample images are not included in the GitHub repository due to size constraints. You can retrain the models using the class.ipynb notebook.







