import math
import numpy as np

class gat_interpreter():
    def __init__(self):
        self.node_level_attention_prev = None
        self.entropy_scores_prev = None
        self.gat_embeddings_prev = None

    # -----------------------------------------------------------
    # Node Attention Map
    # -----------------------------------------------------------
    def build_node_attention_map(self, att_list): # att_weights: the attention weights per time window
       
        edge_level_attention = {} # aggregated over heads
        node_level_attention = {}

        # att_list: [ [(edge_index, alpha), ...], [(edge_index, alpha), ...], ... ]
        for att_batch in att_list:
            if att_batch is None:
                continue

            if isinstance(att_batch, tuple) and len(att_batch) == 2:
                iterator = [att_batch]
            else:
                iterator = att_batch

            for pair in iterator:
                try:
                    edge_index, alpha = pair
                except Exception:
                    continue

                src, dst = edge_index
                try:
                    if hasattr(src, "cpu"):
                        src = src.cpu().numpy()
                    elif hasattr(src, "numpy"):
                        src = src.numpy()
                    else:
                        src = np.array(src)

                    if hasattr(dst, "cpu"):
                        dst = dst.cpu().numpy()
                    elif hasattr(dst, "numpy"):
                        dst = dst.numpy()
                    else:
                        dst = np.array(dst)
                except Exception:
                    src = np.array(src)
                    dst = np.array(dst)

                try:
                    if hasattr(alpha, "detach"):
                        alpha_arr = alpha.detach().cpu().numpy()
                    elif hasattr(alpha, "cpu") and hasattr(alpha, "numpy"):
                        alpha_arr = alpha.cpu().numpy()
                    elif hasattr(alpha, "numpy"):
                        alpha_arr = alpha.numpy()
                    else:
                        alpha_arr = np.array(alpha)
                except Exception:
                    import numpy as np
                    alpha_arr = np.array(alpha)

                if alpha_arr.ndim == 2:
                    if alpha_arr.shape[0] == len(src):
                        alpha_mean = alpha_arr.mean(axis=1)
                    else:
                        alpha_mean = alpha_arr.mean(axis=0)
                else:
                    alpha_mean = alpha_arr.flatten()

                for k in range(len(src)):
                    i = int(src[k])
                    j = int(dst[k])
                    edge = (i, j)
                    val = float(alpha_mean[k])
                    if edge not in edge_level_attention:
                        edge_level_attention[edge] = []
                    edge_level_attention[edge].append(val)

        aggregated_edge_level = {key: float(sum(val)/len(val)) for key, val in edge_level_attention.items()} if edge_level_attention else {}

        for key, val in aggregated_edge_level.items():
            i = key[0]
            j = key[1]
            if i not in node_level_attention:
                node_level_attention[i] = []
            node_level_attention[i].append((j, val))

        return aggregated_edge_level, node_level_attention
    
    
    # -----------------------------------------------------------
    # Attention drift: |α(t) - α(t-1)| summed over all neighbors
    # -----------------------------------------------------------
    def attention_drift(self, node_att_t_prev):
        """
        Compute drift between attention vectors across two time steps.
        """
        drift_scores = {}

        for node in self.node_level_attention:
            # Convert to dict for matching
            at = dict(self.node_level_attention[node])
            ap = dict(node_att_t_prev[node])

            # Union of neighbors
            neighbors = set(at.keys()) | set(ap.keys())

            drift = 0.0
            for j in neighbors:
                drift += abs(at.get(j, 0.0) - ap.get(j, 0.0))

            drift_scores[node] = drift

        total_sum = sum(drift_scores.values())
        if total_sum > 0:
            normalized_drift_scores = {node_id: val / total_sum for node_id, val in drift_scores.items()}
        else:
            normalized_drift_scores = drift_scores


        return normalized_drift_scores
    

    # -----------------------------------------------------------
    # Prediction Error
    # -----------------------------------------------------------
    def prediction_error(self, y_pred, y_true):
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        errors = np.abs(y_pred - y_true)
        # Normalize
        total_error = np.sum(errors)
        pred_errors = {}
        for i in range(len(errors)):
             if total_error > 0:
                 pred_errors[i] = errors[i] / total_error
             else:
                 pred_errors[i] = 0.0 
                 
        if total_error == 0:
             n = len(errors)
             pred_errors = {i: 1.0/n for i in range(n)}
             
        return pred_errors

    # -----------------------------------------------------------
    # Final fusion score for FDI detection
    # -----------------------------------------------------------
    def compute_fdi_score(self, drift_scores, pred_errors=None,
                        w1=0.6, w4=0.4
                        ):
        """
        Simple anomaly fusion:
        S = w1 * Drift + w2 * |ΔEntropy| + w3 * KL + w4 * PredictionError
        """
        fdi_score = {}
        
        if pred_errors is None:
            pred_errors = {node: 0.0 for node in drift_scores}

        for node in drift_scores:
            d = drift_scores[node]

            pe = pred_errors.get(node, 0.0)
            
            score = w1 * d + w4 * pe
            fdi_score[node] = score

        total_sum = sum(fdi_score.values())
        if total_sum > 0:
            normalized_scores = {node_id: val / total_sum for node_id, val in fdi_score.items()}
        else:
            normalized_scores = fdi_score
        return normalized_scores
    # -----------------------------------------------------------
    # embeddings_drift
    # -----------------------------------------------------------
    def embeddings_drift(self, embeddings_t, embeddings_t_prev):
        if embeddings_t_prev is None:
            return np.zeros(embeddings_t.shape[0])
        return np.linalg.norm(embeddings_t - embeddings_t_prev, axis=1)
    # -----------------------------------------------------------
    # __call__
    # -----------------------------------------------------------
    def __call__(self, state):
        att_list = state.get("gat_att") 
        self.att = att_list[0] if len(att_list) > 0 else None
        
        edge_att , self.node_level_attention = self.build_node_attention_map(self.att)
        entropy_scores = self.attention_entropy()

        node_att_prev = state.get("node_level_attention_prev")
        entropy_scores_prev = state.get("entropy_scores_prev")

        if node_att_prev is None:
            node_att_prev = self.node_level_attention_prev

        if entropy_scores_prev is None:
            entropy_scores_prev = self.entropy_scores_prev

        if node_att_prev is None:
            # No drift or KL possible
            drift_scores = {n: 0.0 for n in self.node_level_attention}

            entropy_scores_prev = {n: entropy_scores[n] for n in self.node_level_attention}
        else:
            drift_scores = self.attention_drift(node_att_prev)
        
        y_pred = state.get("y_pred")
        y_true = state.get("y_true")
        
        if y_pred is not None and y_true is not None:
            pred_errors = self.prediction_error(y_pred, y_true)
        else:

            n = len(self.node_level_attention)
            pred_errors = {i: 1.0/n for i in range(n)}

        fdi_score = self.compute_fdi_score(drift_scores=drift_scores, 
                                           entropy_scores=entropy_scores, 
                                           entropy_prev=entropy_scores_prev,
                                           kl_scores=kl_scores,
                                           pred_errors=pred_errors
                                        )
        self.node_level_attention_prev = self.node_level_attention
        self.entropy_scores_prev = entropy_scores
        return {**state,
                "node_level_attention_prev": self.node_level_attention, 
                "entropy_scores_prev": entropy_scores,
                "fdi_score": fdi_score,
                }
