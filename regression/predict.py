#!/usr/bin/env python 

import pandas as pd 
import pickle 
import tensorflow as tf
from tqdm import tqdm

pred_df = pd.read_csv('data/hideen_test.csv') 

with open('scaler.pkl', 'rb') as f:
    scaler_uploaded = pickle.load(f)

scaled_array = scaler_uploaded.transform(pred_df)
scaled_df = pd.DataFrame(scaled_array, columns = pred_df.columns) 

model = tf.keras.models.load_model('results/keras_model.keras')

predictions = []

for index, row in tqdm(scaled_df.iterrows(), total=len(scaled_df)):
    row_features = row.values.reshape(1, -1)
    prediction = model.predict(row_features, verbose=0)
    predictions.append(prediction[0])

predicted_df = pd.DataFrame(predictions, columns=['y_pred'])
result_df = pd.merge(pred_df, predicted_df, left_index=True, right_index=True)

result_df.to_csv('results/final_preds.csv')

print('done')