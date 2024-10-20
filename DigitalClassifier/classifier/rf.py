import pickle 
import cv2 
import matplotlib.pyplot as plt 

def rf_prediction(image_path):
    with open('models/mnist_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(-1, 28 * 28).astype('float32') / 255

    predicted_class = model.predict(img)
    predicted_class = predicted_class.argmax()
    print("Predicted class:", predicted_class)

    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()
