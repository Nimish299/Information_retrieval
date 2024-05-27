import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random
def read_csv(file_path): 
    data = pd.read_csv(file_path, sep=',', header=None, names=['query_id', 'doc_id', 'bm25_score'])
    return data

def read_merged_qrel(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['query_id', 'unused', 'doc_id', 'relevance'])
    return data


def preprocess_data(data):

    data = data.drop(columns=['unused'])
    return data

def generate_pairs(group):
    pairs = []
    total_pairs = len(group) * (len(group) - 1) // 2
    num_pairs_to_keep = int(0.5 * total_pairs)
    selected_pairs = random.sample(range(total_pairs), num_pairs_to_keep)
    pair_index = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            doc_i = group.iloc[i]
            doc_j = group.iloc[j]
            bm25_score_i = doc_i['bm25_score']
            bm25_score_j = doc_j['bm25_score']
            relevance_i = doc_i['relevance']
            relevance_j = doc_j['relevance']
            label = 0.5
           
            if pd.isna(relevance_i) or pd.isna(relevance_j):
                label = 0.5  
            elif relevance_i < relevance_j:
                label = 0  
            elif relevance_i > relevance_j:
                label = 1  
            pairs.append((bm25_score_i, bm25_score_j, label))
    return pairs

csv_file = "test_bm25data1.csv"  
tsv_data = read_csv(csv_file)  

merged_qrel_file = "nfcorpus/merged.qrel"  

merged_qrel_data = read_merged_qrel(merged_qrel_file)
merged_qrel_data = preprocess_data(merged_qrel_data)


merged_data = pd.merge(tsv_data, merged_qrel_data, on=['query_id', 'doc_id'])

grouped_data = merged_data.groupby('query_id')


model_pairwise = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),  
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
model_pairwise.compile(optimizer='adam', loss='binary_crossentropy') 

mse_per_query = {}

test_all_mse = []

for query_id, group in grouped_data:
 
    pairs = generate_pairs(group)
     
    if not pairs:
        continue
    pairwise_df = pd.DataFrame(pairs, columns=['bm25_score_i', 'bm25_score_j', 'label'])

   
    X_pairwise = pairwise_df[['bm25_score_i', 'bm25_score_j']].values
    y_pairwise = pairwise_df['label'].values

    
    scaler_pairwise = StandardScaler()
    X_pairwise_scaled = scaler_pairwise.fit_transform(X_pairwise)


    X_pairwise_scaled = scaler_pairwise.fit_transform(X_pairwise)
    X_pairwise_scaled_reshaped = X_pairwise_scaled.reshape(-1, 2)  
    history_pairwise = model_pairwise.fit(X_pairwise_scaled_reshaped, y_pairwise, epochs=2, batch_size=64)

    y_pred_pairwise = model_pairwise.predict(X_pairwise_scaled)
    mse_pairwise = mean_squared_error(y_pairwise, y_pred_pairwise)
    test_all_mse.append(mse_pairwise)
  

average_test_mse = np.mean(test_all_mse)
print('Average Mean Squared Error on Testing Data:', average_test_mse)