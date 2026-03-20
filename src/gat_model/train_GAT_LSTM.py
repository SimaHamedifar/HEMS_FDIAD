import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os
import logging
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import get_dir
from src.data_processing.scale import scale
from src.data_processing.sliding_window_graph_data import sliding_window_graph_data

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.gat_model.GAT_LSTM import GAT_LSTM_Forecaster

def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def training_criterion(predictions, targets):
    # predictions/targets shape: (batch_size, num_nodes)
    # Assume net_load is the last node (index 4)
    
    # Calculate individual MSE for each node
    mse_per_node = torch.mean((predictions - targets)**2, dim=0)
    
    # Apply higher weight to the net_load (index 4)
    # Weights for: [shiftable, base, demand, generation, net_load]
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0]).to(predictions.device)
    
    weighted_loss = torch.sum(mse_per_node * weights)
    return weighted_loss

def train(args):
    # logging
    setup_logging(args.log_file)
    logging.info(f"Starting training with args: {args}")

    # Scale the data. 
    try:
        logging.info("Scaling info ...")
        train_scaled, val_scaled, test_scaled, node_idx, context_idx = scale()
        logging.info(f"Data is scaled. shape of train data: {train_scaled.shape}, shape of validation data: {val_scaled.shape}, shape of test data: {test_scaled.shape}")
    except Exception as e:
        logging.info(f"Error during Scaling: {e}", exc_info=True)
        return
    
    # Slice the data. 
    try:
        logging.info("Preparing graph data by a sliding window ....")
        X_train, y_train = sliding_window_graph_data(data_array=train_scaled, window_size=args.window_size, node_indices=node_idx, context_indices=context_idx)
        X_val, y_val = sliding_window_graph_data(data_array=val_scaled, window_size=args.window_size, node_indices=node_idx, context_indices=context_idx)
        X_test, y_test = sliding_window_graph_data(data_array=test_scaled, window_size=args.window_size, node_indices=node_idx, context_indices=context_idx)
        logging.info(f"The data is prepared for training the GAT_LSTM model. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}.")
        logging.info(f"The data is prepared for validating the training of the GAT_LSTM model. X_val shape: {X_val.shape}, y_val shape: {y_val.shape}.")
        logging.info(f"The data is prepared for testing the GAT_LSTM model. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}.")
    except Exception as e:
        logging.info(f"Error during preparing the data using a sliding window: {e}", exc_info=True)

    # Create DataLoaders. 
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)

    # Initialize the model.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    num_nodes = len(node_idx)
    in_channels = 1 + len(context_idx)
    out_channels = args.hidden_dim
    correlation_matrix = np.array([[ 1.        ,  0.2550236 ,  0.27469072, -0.16272809,  0.28558874],
       [ 0.2550236 ,  1.        ,  0.99916837,  0.27504591,  0.99579799],
       [ 0.27469072,  0.99916837,  1.        ,  0.2671857 ,  0.99700915],
       [-0.16272809,  0.27504591,  0.2671857 ,  1.        ,  0.21279867],
       [ 0.28558874,  0.99579799,  0.99700915,  0.21279867,  1.        ]])
    model = GAT_LSTM_Forecaster(num_nodes=num_nodes,
                                in_channels=in_channels,
                                out_channels=out_channels, 
                                correlation_matrix = correlation_matrix,
                                heads=args.heads,
                                dropout=args.dropout
                                ).to(device)
    logging.info("Initialized the model. ")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    # criterion = training_criterion

    epochs = args.epochs
    if args.dry_run:
        epochs = 1
        logging.info("Dry run mode enabled. Training for 1 epoch.")
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_list.append(val_loss)

        logging.info(f"Epoch: {epoch+1}: Training Loss: {train_loss: .4f}, Validation Loss: {val_loss: .4f}")
        
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            if not args.dry_run:
                torch.save(model.state_dict(), get_dir("models") / 'best_model.pth')
                logging.info("New model saved to best_model.pth.")

    logging.info(f"Training is completed. Best Validation Loss = {best_val_loss: .4f}")
    return np.array(train_loss_list), np.array(val_loss_list)