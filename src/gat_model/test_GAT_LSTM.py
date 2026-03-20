import torch
import numpy as np
import pandas as pd
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import get_dir
from src.data_processing.scale import scale
from src.data_processing.sliding_window_graph_data import sliding_window_graph_data
from src.gat_model.GAT_LSTM import GAT_LSTM_Forecaster
from src.gat_model.gat_interpreter import gat_interpreter

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Constants
    WINDOW_SIZE = 30
    BATCH_SIZE = 1
    HIDDEN_DIM = 16
    MODEL_PATH = get_dir("models") / "best_model.pth"
    THRESHOLD_FILE = get_dir("outputs") / "fdi_threshold.txt"
    # Correlation matrix from train_GAT_LSTM.py / run_analysis.py
    CORRELATION_MATRIX = np.array([
       [ 1.        ,  0.2550236 ,  0.27469072, -0.16272809,  0.28558874],
       [ 0.2550236 ,  1.        ,  0.99916837,  0.27504591,  0.99579799],
       [ 0.27469072,  0.99916837,  1.        ,  0.2671857 ,  0.99700915],
       [-0.16272809,  0.27504591,  0.2671857 ,  1.        ,  0.21279867],
       [ 0.28558874,  0.99579799,  0.99700915,  0.21279867,  1.        ]
    ])

    # Setup Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    # ==========================
    # Validation Phase (Threshold)
    # ==========================
    logging.info("Starting Validation Phase to compute Threshold...")
    
    # Load and Scale Data
    # mode='non_attacked' loads train/val/test from standard files and saves scaler
    _, val_scaled, _, node_idx, ctx_idx = scale(mode='non_attacked')
    
    # Prepare Windows
    X_val, y_val = sliding_window_graph_data(val_scaled, WINDOW_SIZE, node_idx, ctx_idx)
    
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    num_nodes = len(node_idx)
    in_channels = 1 + len(ctx_idx)
    model = GAT_LSTM_Forecaster(
        num_nodes=num_nodes,
        in_channels=in_channels,
        out_channels=HIDDEN_DIM,
        correlation_matrix=CORRELATION_MATRIX,
        heads=2,
        dropout=0.2
    ).to(device)
    
    # Load Weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logging.info(f"Loaded model from {MODEL_PATH}")
    else:
        logging.error(f"Model file {MODEL_PATH} not found!")
        return

    model.eval()
    interpreter = gat_interpreter()
    interpreter_state = {
        "node_level_attention_prev": None,
        "entropy_scores_prev": None
    }
    
    max_fdi_score = -float('inf')
    
    logging.info("Iterating through validation data...")
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device)
            # target is (1, Nodes)
            
            # Forward
            pred, attn_weights, edge_index_out = model(data, return_att=True)
            # attn_weights: (Batch, Seq, Edges, Heads)
            # Last step attention: (1, Num_Edges, Heads)
            last_step_alpha = attn_weights[0, -1, :, :] 
            
            gat_att_input = [ [(model.edge_index, last_step_alpha)] ]
            
            interpreter_state["gat_att"] = gat_att_input
            interpreter_state["y_pred"] = pred.cpu().numpy()[0]
            interpreter_state["y_true"] = target.numpy()[0]
            
            new_state = interpreter(interpreter_state)
            interpreter_state = new_state
            
            fdi_scores = new_state["fdi_score"] # dict
            current_max = max(fdi_scores.values())
            
            if current_max > max_fdi_score:
                max_fdi_score = current_max
                
    threshold = max_fdi_score
    logging.info(f"Computed Threshold (Max FDI Score on Validation): {threshold}")
    
    with open(THRESHOLD_FILE, "w") as f:
        f.write(str(threshold))
    logging.info(f"Threshold saved to {THRESHOLD_FILE}")
    
    # ==========================
    # Test Phase (Detection)
    # ==========================
    logging.info("Starting Test Phase (Detection) on Data/df_test_attacked_0.1.csv...")
    
    test_file = get_dir("data") / "df_test_attacked_0.1.csv"
    if not os.path.exists(test_file):
        logging.error(f"Test file {test_file} not found!")
        return
        
    df_test = pd.read_csv(test_file)
    
    # Extract Ground Truth
    labels_full = df_test["is_attacked"].values
    
    # Drop columns for scaling
    cols_to_drop = ['time', 'hour', 'is_attacked']
    df_clean = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
    
    # Scale
    # mode='attacked' uses the saved scaler from validation phase
    test_scaled, _, _ = scale(mode='attacked', window_data=df_clean)
    
    # Make windows
    X_test, y_test = sliding_window_graph_data(test_scaled, WINDOW_SIZE, node_idx, ctx_idx)
    
    # Align labels
    # Target y[i] is at index i + WINDOW_SIZE in original data
    labels_aligned = labels_full[WINDOW_SIZE:]
    
    if len(labels_aligned) != len(X_test):
        logging.error(f"Shape mismatch: X_test {len(X_test)}, labels {len(labels_aligned)}")
        return

    test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Reset interpreter
    interpreter = gat_interpreter()
    interpreter_state = {
        "node_level_attention_prev": None,
        "entropy_scores_prev": None
    }
    
    predictions = []
    
    logging.info("Iterating through test data...")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            
            pred, attn_weights, edge_index_out = model(data, return_att=True)
            last_step_alpha = attn_weights[0, -1, :, :] 
            
            gat_att_input = [ [(model.edge_index, last_step_alpha)] ]
            
            interpreter_state["gat_att"] = gat_att_input
            interpreter_state["y_pred"] = pred.cpu().numpy()[0]
            interpreter_state["y_true"] = target.numpy()[0]
            
            new_state = interpreter(interpreter_state)
            interpreter_state = new_state
            
            fdi_scores = new_state["fdi_score"]
            current_max = max(fdi_scores.values())
            
            is_attack = 1 if current_max > threshold else 0
            predictions.append(is_attack)
            
    # Metrics
    acc = accuracy_score(labels_aligned, predictions)
    cm = confusion_matrix(labels_aligned, predictions)
    report = classification_report(labels_aligned, predictions)
    
    logging.info(f"Detection Accuracy: {acc:.4f}")
    logging.info("Confusion Matrix:")
    logging.info(f"\n{cm}")
    logging.info("Classification Report:")
    logging.info(f"\n{report}")
    
if __name__ == "__main__":
    main()
