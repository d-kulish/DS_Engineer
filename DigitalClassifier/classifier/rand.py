import cv2 
import numpy as np 
import matplotlib.pyplot as plt 


def rand_prediction(self, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
        
    predicted_classes = np.random.randint(0, 10)
    print("Predicted class:", predicted_classes)

    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()
