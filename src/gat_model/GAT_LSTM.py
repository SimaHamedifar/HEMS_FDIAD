import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import dense_to_sparse

class GAT_LSTM_Forecaster(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, correlation_matrix, heads=2, dropout=0.2):
        super().__init__()
        adj = torch.abs(torch.tensor(correlation_matrix, dtype=torch.float))
        self.edge_index, self.edge_weight = dense_to_sparse(adj)

        self.gat = GATv2Conv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout, edge_dim=1,
            add_self_loops=True)
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=64, batch_first=True)
        self.decoder = nn.Linear(in_features=64, out_features=1) # predict 1 value per node
        # self.edge_index = self._create_fully_connected_edges(num_nodes)

    # def _create_fully_connected_edges(self, num_nodes):
    #     adj = torch.ones((num_nodes, num_nodes))
    #     edge_index, _ = dense_to_sparse(adj)
    #     return edge_index
    
    def forward(self, x, return_att=False):
        # x: (Batch, Seq_Len, Nodes, Features)
        batch_size, seq_len, num_nodes, num_features = x.size()
        # Use the device of the input tensor to avoid device mismatches
        device = x.device
        # Do not move `x` here; caller should place model and inputs on the same device.

        # 1. Flatten Batch and Seq into one dimension for the GAT
        # New shape: (Batch * Seq_Len * Nodes, Features)
        x_flat = x.view(-1, num_features)

        # 2. Create a "Batch" of edges
        # We need to repeat our edge_index for every (batch * seq) instance
        # This allows GAT to process all time steps and batches in parallel
        num_graphs = batch_size * seq_len
        
        # Efficiently create batched edge_index
        # Each graph's edge_index is offset by (graph_index * num_nodes)
        edge_index = self.edge_index.to(device)
        offsets = torch.arange(num_graphs, device=device).view(-1, 1, 1) * num_nodes
        batched_edge_index = (edge_index.unsqueeze(0) + offsets).transpose(0, 1).reshape(2, -1)

        # 3. Apply GAT to all nodes at once
        # gat_out shape: (Batch * Seq_Len * Nodes, out_channels)
        if return_att:
            gat_out, (edge_index_out, alpha) = self.gat(x_flat, batched_edge_index, return_attention_weights=True)
        else:
            gat_out = self.gat(x_flat, batched_edge_index)

        # 4. Reshape for LSTM
        # We want: (Batch * Nodes, Seq_Len, out_channels)
        gat_out = gat_out.view(batch_size, seq_len, num_nodes, -1)
        lstm_input = gat_out.permute(0, 2, 1, 3).contiguous() # (Batch, Nodes, Seq, Hidden)
        lstm_input = lstm_input.view(batch_size * num_nodes, seq_len, -1)

        # 5. LSTM and Decoder
        lstm_output, _ = self.lstm(lstm_input)
        lstm_last = lstm_output[:, -1, :] # (Batch * Nodes, 64)
        
        # 6. Final MTL Prediction
        # decoder_output: (Batch * Nodes, 1)
        decoder_output = self.decoder(lstm_last) 
        
        # Reshape back to (Batch, Nodes)
        prediction = decoder_output.view(batch_size, num_nodes)
        if return_att:
            # alpha: (Batch * Seq * Num_Edges, Heads)
            #  change to (Batch, Seq, Num_Edges, Heads)
            num_edges = self.edge_index.size(1)
            alpha = alpha.view(batch_size, seq_len, num_edges, -1)
            return prediction, alpha, edge_index_out
        
        return prediction