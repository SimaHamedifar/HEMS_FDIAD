import torch
import numpy as np
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.utils.paths import get_dir
from src.env.HEMS_env import HEMSenv as SmartGridEnv
from src.agents.lstm_sac_main import MLPActorCritic

def get_action(ac, o, o_buff, a_buff, o_buff_len, act_dim, act_limit, deterministic=False, device=None):
    h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
    h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
    h_l = torch.tensor([o_buff_len]).float().to(device)
    with torch.no_grad():
        a = ac.act(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(device),
                   h_o, h_a, h_l, deterministic=deterministic, device=device).reshape(act_dim)
    return np.clip(a, -act_limit, act_limit)

def run_test():
    max_hist_len = 5
    
    critic_mem_pre_lstm_hid_sizes = (128,)
    critic_mem_lstm_hid_sizes = (128,)
    critic_mem_after_lstm_hid_size = ()
    critic_cur_feature_hid_sizes = (128, 128)
    critic_post_comb_hid_sizes = (128,)
    critic_hist_with_past_act = False
    
    actor_mem_pre_lstm_hid_sizes = (128,)
    actor_mem_lstm_hid_sizes = (128,)
    actor_mem_after_lstm_hid_size = ()
    actor_cur_feature_hid_sizes = (128, 128)
    actor_post_comb_hid_sizes = (128,)
    actor_hist_with_past_act = True
    
    # Path to checkpoint
    checkpoint_path = get_dir("models") / 'checkpoint-model-Step-99999.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Initialize Environment with Test Data
    data_path = get_dir("data") / 'df_test_attacked_0.1.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    print(f"Initializing SmartGridEnv with {data_path}...")
    # Note: SmartGridEnv might need best_model.pth which is in root
    env = SmartGridEnv(data_path=data_path)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    
    # Initialize Model
    ac = MLPActorCritic(obs_dim, act_dim, act_limit,
                        critic_mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                        critic_mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                        critic_mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                        critic_cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                        critic_post_comb_hid_sizes=critic_post_comb_hid_sizes,
                        critic_hist_with_past_act=critic_hist_with_past_act,
                        actor_mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                        actor_mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                        actor_mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                        actor_cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                        actor_post_comb_hid_sizes=actor_post_comb_hid_sizes,
                        actor_hist_with_past_act=actor_hist_with_past_act)
    
    # Load Checkpoint
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # For compatibility, try mps if available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Loading checkpoint from {checkpoint_path} to {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ac.load_state_dict(checkpoint['ac_state_dict'])
    ac.to(device)
    ac.eval()
    
    print(f"Starting testing...")
    
    # One episode covers the whole dataset in SmartGridEnv
    res = env.reset()
    if isinstance(res, tuple):
        o = res[0]
    else:
        o = res
            
    ep_ret = 0
    ep_len = 0
    
    # Initialize history buffer
    if max_hist_len > 0:
        o_buff = np.zeros([max_hist_len, obs_dim])
        a_buff = np.zeros([max_hist_len, act_dim])
        o_buff[0, :] = o
        o_buff_len = 0
    else:
        o_buff = np.zeros([1, obs_dim])
        a_buff = np.zeros([1, act_dim])
        o_buff_len = 0
    
    done = False
    
    y_true = []
    y_pred = []
    
    while not done:
        a = get_action(ac, o, o_buff, a_buff, o_buff_len, act_dim, act_limit, deterministic=True, device=device)
        
        # Env step
        res = env.step(a)
        
        # Handle Gym version differences
        if len(res) == 5:
            o2, r, terminated, truncated, info = res
            done = terminated or truncated
        elif len(res) == 4:
            o2, r, done, info = res
        else:
            # Fallback for old gym if info is missing? No, SmartGridEnvNew returns 4 values including info
            raise ValueError(f"Unexpected step return length: {len(res)}")
        
        ep_ret += r
        ep_len += 1
        
        # Collect Metrics
        
        # Get ground truth from env directly because info['is_attack'] returns 'N/A' after 960 steps
        # env.current_step is the index of the step that will be predicted NEXT, 
        # but the step() we just called processed the prediction for the previous step.
        # SmartGridEnv logic:
        # self.current_step starts at window_size.
        # step() uses self.is_attacked[self.current_step]
        # then self.current_step += 1
        # So we need the index that was used in step(). That is env.current_step - 1.
        
        current_idx = env.current_step - 1
        if current_idx < len(env.is_attacked):
            is_attack_truth = env.is_attacked.iloc[current_idx-2] if hasattr(env.is_attacked, 'iloc') else env.is_attacked[current_idx-2]
            
            # detected comes from info['detected'] which is computed correctly in step()
            detected = info.get('detected')
            
            # Ensure we have boolean or int
            y_true.append(int(is_attack_truth))
            y_pred.append(int(detected))
        
        # Update history
        if max_hist_len != 0:
            if o_buff_len == max_hist_len:
                o_buff[:max_hist_len - 1] = o_buff[1:]
                a_buff[:max_hist_len - 1] = a_buff[1:]
                o_buff[max_hist_len - 1] = list(o)
                a_buff[max_hist_len - 1] = list(a)
            else:
                o_buff[o_buff_len + 1 - 1] = list(o)
                a_buff[o_buff_len + 1 - 1] = list(a)
                o_buff_len += 1
        
        o = o2
        
    print("-" * 30)
    print(f"Test finished.")
    print(f"Total Reward: {ep_ret:.2f}")
    print(f"Steps: {ep_len}")
    
    # Calculate Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) == 0:
        print("No data collected for metrics.")
        return

    cm = confusion_matrix(y_true, y_pred, )
    # confusion_matrix returns [[TN, FP], [FN, TP]]
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # Only one class present
        if y_true[0] == 0:
            tn, fp, fn, tp = cm[0,0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0,0]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0) # TPR
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    print("-" * 30)
    print("Confusion Matrix:")
    print(cm)
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive Rate (Recall): {recall:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")
    print(f"Precision: {precision:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    run_test()
