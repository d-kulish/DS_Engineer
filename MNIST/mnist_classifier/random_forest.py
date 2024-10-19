from sklearn.ensemble import RandomForestClassifier
from mnist_classifier.utils import download_mnist

def create_model(): 
  X_train, y_train, X_test, y_test = download_mnist()
  X_train = X_train.reshape(-1, 784)
  X_test = X_test.reshape(-1, 784)
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train) 
  
  return model