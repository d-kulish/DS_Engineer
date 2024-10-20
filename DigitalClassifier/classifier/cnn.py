import cv2 
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
import numpy as np 

def cnn_prediction(image_path): 
    model = load_model('models/mnist_model.keras')
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(-1, 28, 28, 1).astype('float32') / 255

    predicted_class = model.predict(img).argmax()
    print("Predicted class:", predicted_class)

    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()
