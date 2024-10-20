import os
import random
from classifier import cnn_prediction, rf_prediction, rand_prediction

def get_user_choice():
    while True:
        choice = input("Choose an algorithm (rf for Random Forest, cnn for Convolutional or rand for Random): ")
        if choice.lower() in ["rf", "cnn", "rand"]:
            return choice.lower()
        else:
            print("Invalid choice. Please enter 'rf', 'cnn', or 'rand'.")

def load_image(image_folder):
    images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]  
    if not images:
        print("No images found in the specified folder.")
        return None
    random_image = random.choice(images)
    image_path = os.path.join(image_folder, random_image)
    return image_path

def make_prediction(algorithm, image_path):
    if algorithm == "rf":
        predictor = rf_prediction(image_path)
    elif algorithm == "cnn":
        predictor = cnn_prediction(image_path)
    elif algorithm == "rand":
        predictor = rand_prediction(image_path)
    else:
        raise ValueError("Invalid algorithm choice.")

    print(image_path)
    
    if predictor is not None:
        predictor.predict()
    else:
        print("Error: Predictor object is None.")

def main():
    image_folder = "images"  
    algorithm = get_user_choice()
    image_path = load_image(image_folder)

    if image_path:
        make_prediction(algorithm, image_path)

if __name__ == "__main__":
    main()