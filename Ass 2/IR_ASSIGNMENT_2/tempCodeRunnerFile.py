# Pairwise Training and Evaluation
for query_id, group in grouped_data:
    # Pair Generation
    pairs = generate_pairs(group)
     # Check if there are pairs for this query ID
    if not pairs:
        continue
    pairwise_df = pd.DataFrame(pairs, columns=['bm25_score_i', 'bm25_score_j', 'label'])

    # Split features and labels
    X_pairwise = pairwise_df[['bm25_score_i', 'bm25_score_j']].values
    y_pairwise = pairwise_df['label'].values

    # Standardize features
    scaler_pairwise = StandardScaler()
    X_pairwise_scaled = scaler_pairwise.fit_transform(X_pairwise)

    # Train pairwise model
    # history_pairwise = model_pairwise.fit(X_pairwise_scaled, y_pairwise, epochs=2,batch_size=64)
    X_pairwise_scaled = scaler_pairwise.fit_transform(X_pairwise)
    X_pairwise_scaled_reshaped = X_pairwise_scaled.reshape(-1, 2)  # Reshape input data
    history_pairwise = model_pairwise.fit(X_pairwise_scaled_reshaped, y_pairwise, epochs=2, batch_size=64)
    # Compute MSE on validation data
    y_pred_pairwise = model_pairwise.predict(X_pairwise_scaled)
    mse_pairwise = mean_squared_error(y_pairwise, y_pred_pairwise)
    test_all_mse.append(mse_pairwise)
    # Store MSE for the query ID
    # mse_per_query[query_id] = mse_pairwise