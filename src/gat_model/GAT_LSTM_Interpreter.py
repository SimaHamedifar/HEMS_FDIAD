import numpy as np
import torch
import os

from src.utils.paths import get_dir
from src.data_processing.scale import scale
from src.gat_model.GAT_LSTM import GAT_LSTM_Forecaster
from src.gat_model.gat_interpreter import gat_interpreter

class GAT_LSTM_Interpreter():
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = get_dir("models") / 'best_model.pth'
        self.correlation_matrix = np.array([
           [ 1.        ,  0.2550236 ,  0.27469072, -0.16272809,  0.28558874],
           [ 0.2550236 ,  1.        ,  0.99916837,  0.27504591,  0.99579799],
           [ 0.27469072,  0.99916837,  1.        ,  0.2671857 ,  0.99700915],
           [-0.16272809,  0.27504591,  0.2671857 ,  1.        ,  0.21279867],
           [ 0.28558874,  0.99579799,  0.99700915,  0.21279867,  1.        ]
        ])
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    
        # Columns: ['time_sin', 'time_cos', 'is_weekend', 'day_of_week', 'shiftable_loads', 'base_loads', 'generation', 'demand', 'net_load']
        # nodes: ['shiftable_loads', 'base_loads', 'demand', 'generation', 'net_load']
        self.node_idx = [4, 5, 7, 6, 8]
        # ctx: ['time_sin', 'time_cos', 'is_weekend', 'day_of_week']
        self.ctx_idx = [0, 1, 2, 3]
        
        num_nodes = len(self.node_idx)
        in_channels = 1 + len(self.ctx_idx)
        out_channels = 16 # hidden_dim

        self.model = GAT_LSTM_Forecaster(
            num_nodes=num_nodes,
            in_channels=in_channels,
            out_channels=out_channels,
            correlation_matrix=self.correlation_matrix,
            heads=2,
            dropout=0.2
        ).to(self.device)

        if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: {model_path} not found. Using initialized weights.")
        
        self.model.eval()

        self.interpreter = gat_interpreter()


    def __call__(self, obs_df, interpreter_state): 
        scaled_data, nodes_idx, ctx_idx = scale(mode='attacked', 
                                                window_data=obs_df)
        num_nodes = len(self.node_idx)
        num_ctx = len(self.ctx_idx)
        
        node_feats = scaled_data[:, self.node_idx] # (30, 5)
        
        ctx_feats = scaled_data[:, self.ctx_idx] # (30, 4)

        ctx_expanded = np.repeat(ctx_feats[:, np.newaxis, :], num_nodes, axis=1)
        x_input = np.concatenate([node_feats[:, :, np.newaxis], ctx_expanded], axis=2)
        x_tensor = torch.FloatTensor(x_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds, attn_weights, edge_index_out = self.model(x_tensor, return_att=True)
        
        # Extract attention for the last time step
        # attn_weights shape: (Batch, Seq, Num_Edges, Heads) -> (1, 24, Num_Edges, Heads)
        last_step_alpha = attn_weights[0, -1, :, :] # (Num_Edges, Heads)
        
        
        gat_att_input = [ [(self.model.edge_index, last_step_alpha)] ]
        
        # Interpreter Step
        interpreter_state["gat_att"] = gat_att_input
        
        # Call interpreter
        interpreter_state = self.interpreter(interpreter_state)
        
        fdi_scores = interpreter_state["fdi_score"] # Dict {node_idx: score}

        return fdi_scores
        
        
        
        

