#!/usr/bin/env python

import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import pickle 
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tqdm.keras import TqdmCallback


train_df = pd.read_csv('data/train.csv')

scaler = MinMaxScaler() 

target = train_df.pop('target')

with open('results/scaler.pkl', 'rb') as f:
    scaler_uploaded = pickle.load(f)

scaled_array = scaler_uploaded.transform(train_df)
scaled_df = pd.DataFrame(scaled_array, columns = train_df.columns) 

X_train, X_test, y_train, y_test = train_test_split(scaled_df, target, test_size=0.2, random_state=42)

model_keras = Sequential()
model_keras.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))  
model_keras.add(Dense(units=32, activation='relu'))  
model_keras.add(Dense(units=1, activation='linear'))  

model_keras.compile(optimizer='adam', loss='mean_squared_error')

tqdm_callback = TqdmCallback(total=len(X_train) // 32)
model_keras.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, callbacks=[tqdm_callback])

predictions = model_keras.predict(X_test, verbose = 0)

mse_keras = mean_squared_error(y_test, predictions)
r2_keras = r2_score(y_test, predictions)

model_keras.save('results/keras_model.keras')

result_dict = {
    'mse': mse_keras, 
    'r2': r2_keras
}

print(result_dict)

with open('results/train_results.json', 'w') as f:
    json.dump(result_dict, f)

print('done')