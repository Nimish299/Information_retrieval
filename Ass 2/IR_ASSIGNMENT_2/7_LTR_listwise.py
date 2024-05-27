import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def read_csv(file_path):
    data = pd.read_csv(file_path, sep=',', header=None, names=['query_id', 'doc_id', 'bm25_score'])
    return data

def read_merged_qrel(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['query_id', 'unused', 'doc_id', 'relevance'])
    return data

def preprocess_data(data):

    data = data.drop(columns=['unused'])
    return data


csv_file = "test_bm25data1.csv"
tsv_file = "nfcorpus/merged.qrel"


tsv_data = read_csv(csv_file)
merged_qrel_data = read_merged_qrel(tsv_file)
merged_qrel_data = preprocess_data(merged_qrel_data)


merged_data = pd.merge(tsv_data, merged_qrel_data, on=['query_id', 'doc_id'])


model_directory = 'saved_model/'
grouped_train_data = merged_data.groupby('query_id')


all_X_train = []
all_y_train = []


for query_id, group in grouped_train_data:
    X_query = group[['bm25_score']].values
    y_query = group['relevance'].values.astype(float) / 3.0  

    
    all_X_train.append(X_query)
    all_y_train.append(y_query)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(None, 1)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  
])


model.compile(optimizer='adam', loss='mse')


for X_query, y_query in zip(all_X_train, all_y_train):
    
    X_query = np.expand_dims(X_query, axis=0)
    y_query = np.expand_dims(y_query, axis=0)

    
    model.fit(X_query, y_query, epochs=2, batch_size=32, verbose=0)  


y_pred_train = np.concatenate([model.predict(np.expand_dims(X_query, axis=0)) for X_query in all_X_train], axis=1)


mse_train = mean_squared_error(np.concatenate(all_y_train), y_pred_train.flatten())
model_filename = 'saved_modelliswise_model.h5'
model.save(model_filename)

print('Mean Squared Error on Training Data:', mse_train)
