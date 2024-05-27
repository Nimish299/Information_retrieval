import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


def read_csv(file_path):
    data = pd.read_csv(file_path, sep=',', header=None, names=['query_id', 'doc_id', 'bm25_score','tf'])
    return data


def read_merged_qrel(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['query_id', 'unused', 'doc_id', 'relevance'])
    return data


def preprocess_data(data):
    
    data = data.drop(columns=['unused'])
    return data

csv_file = "test_bm25data.csv"
tsv_data = read_csv(csv_file)  

merged_qrel_file = "nfcorpus/merged.qrel"
merged_qrel_data = read_merged_qrel(merged_qrel_file)
merged_qrel_data = preprocess_data(merged_qrel_data)


merged_data = pd.merge(tsv_data, merged_qrel_data, on=['query_id', 'doc_id'])


grouped_data = merged_data.groupby('query_id')


all_mse = []
model_directory = 'saved_model/'


X_query = merged_data[['bm25_score', 'tf']].values
scaler_query = StandardScaler()
X_query_scaled = scaler_query.fit_transform(X_query)


model_query = Sequential([
    Dense(64, activation='relu', input_shape=(X_query.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  
])
model_query.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')


for query_id, group in grouped_data:
    drop_indices = np.random.choice(group.index, size=int(0.1 * len(group)), replace=False)
    group = group.drop(drop_indices)
    
    X_train_query = group[['bm25_score', 'tf']].values
    y_train_query = group['relevance'].values.astype(float) / 3.0  

    
    X_train_query_scaled = scaler_query.transform(X_train_query)
    
    
    history_query = model_query.fit(X_train_query_scaled, y_train_query, epochs=2, batch_size=32)

    y_pred_train_query = model_query.predict(X_train_query_scaled)
    mse_train_query = mean_squared_error(y_train_query, y_pred_train_query)
    all_mse.append(mse_train_query)


average_mse = np.mean(all_mse)
print('Average Mean Squared Error:', average_mse)


model_filename = 'saved_model/pointwise_model.h5'
model_query.save(model_filename)


loaded_model = load_model(model_filename)


test_csv_file = "test_bm25data.csv"
test_csv_data = read_csv(test_csv_file)


merged_test_data = pd.merge(test_csv_data, merged_qrel_data, on=['query_id', 'doc_id'])


grouped_test_data = merged_test_data.groupby('query_id')


test_all_mse = []


for query_id, test_group in grouped_test_data:
    X_test_query = test_group[['bm25_score', 'tf']].values
    y_test_query = test_group['relevance'].values.astype(float) / 3.0  

    
    X_test_query_scaled = scaler_query.transform(X_test_query)

    
    y_pred_test_query = loaded_model.predict(X_test_query_scaled)
    mse_test_query = mean_squared_error(y_test_query, y_pred_test_query)
    test_all_mse.append(mse_test_query)


average_test_mse = np.mean(test_all_mse)
print('Average Mean Squared Error on Testing- Data:', average_test_mse)
