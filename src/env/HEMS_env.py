import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os
from src.utils.paths import get_dir

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gat_model.GAT_LSTM_Interpreter import GAT_LSTM_Interpreter

class HEMSenv(gym.Env):
    def __init__(self, window_size=30, hidden_dim=16, model_path=None, w_TP =5, w_FP = 1, w_TN = 1, w_FN = 5):
        super(HEMSenv, self).__init__()
        
        if model_path is None:
            model_path = get_dir("models") / "best_model.pth"
            
        self.window_size = window_size

        self.w_TP = w_TP
        self.w_TN = w_TN
        self.w_FP = w_FP
        self.w_FN = w_FN

        data_path = get_dir("data") / "df_train_attacked_0.1.csv"
        
        print("Loading attacked data...")
        self.df = pd.read_csv(data_path)

        if 'is_attacked' in self.df.columns:
            self.is_attacked = self.df.is_attacked
        else:
            print("Warning: 'is_attacked' column not found. Generating based on hour 8-17.")
            self.df['is_attacked'] = 0
            if 'hour' in self.df.columns:
                self.df.loc[(self.df['hour'] >= 8) & (self.df['hour'] <= 17), 'is_attacked'] = 1

        self.is_attacked = self.df.is_attacked

        self.data = self.df[['time_sin', 'time_cos', 'is_weekend', 'day_of_week', 'shiftable_loads', 'base_loads', 'generation', 'demand', 'net_load']]
        
        self.num_features = self.data.shape[1] # Should be 9
        
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')

        self.gat_lstm_interpreter = GAT_LSTM_Interpreter(model_path=model_path)

        self.interpreter_state = {
            "node_level_attention_prev": None,
            "entropy_scores_prev": None
        }
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size * self.num_features,), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.current_step = 0
        
    def reset(self):
        self.current_step = self.window_size

        self.interpreter_state = {
            "node_level_attention_prev": None,
            "entropy_scores_prev": None
        }
        
        return self._get_obs()
        
    def _get_obs(self):
        start = self.current_step - self.window_size
        end = self.current_step
        obs_window = self.data[start:end]
        return obs_window.values.flatten().astype(np.float32)
    
    
    def step(self, action):
        threshold = action[0]
        
        # 1. Make the current observation.
        start = self.current_step - self.window_size
        end = self.current_step
        window_data = self.data[start:end].copy() 

        fdi_scores = self.gat_lstm_interpreter(obs_df=window_data, 
                                                
                                               interpreter_state=self.interpreter_state)
        max_fdi_score = max(fdi_scores.values()) if fdi_scores else 0.0
        
        detected = max_fdi_score > threshold
        
        #  Consider adding weights to TP, maybe ?!!
        if self.is_attacked[self.current_step-1].item() and detected:
            reward = 1.0 * self.w_TP  # True Positive
        elif not self.is_attacked[self.current_step-1].item() and not detected:
            reward = 1.0 * self.w_TN # True Negative
        elif self.is_attacked[self.current_step-1].item() and not detected:
            reward = -1.0 * self.w_FN # False Negative
        else: # not is_attack and detected
            reward = -1.0 * self.w_FP # False Positive
            
        obs = self._get_obs()
        
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        return obs, reward, done, {}

   