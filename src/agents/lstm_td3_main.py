from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import itertools
import os
import os.path as osp
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.utils.paths import get_dir
from src.env.HEMS_env import HEMSenv

class EpochLogger:
    def __init__(self, output_dir=None, exp_name=None, seed=0, datestamp=False):
        self.output_dir = output_dir or str(get_dir("outputs") / "experiments")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.epoch_dict = {}
        self.log_file = open(os.path.join(self.output_dir, "log.txt"), 'w')
        self.step_log_file = open(os.path.join(self.output_dir, "step_rewards_lstm_TD3.csv"), 'w')
        self.step_log_file.write("step,reward\n")

    def log_step_reward(self, step, reward):
        self.step_log_file.write(f"{step},{reward}\n")
        self.step_log_file.flush()

    def save_config(self, config):
        with open(os.path.join(self.output_dir, "config.json"), 'w') as f:
            json.dump(config, f, default=lambda x: str(x), indent=4)

    def log(self, msg, color=None):
        print(msg)
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.epoch_dict:
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        if val is not None:
            self.log(f"{key}: {val}")
        else:
            vals = self.epoch_dict.get(key, [])
            if vals:
                self.log(f"{key}: {np.mean(vals)}")
                if with_min_and_max:
                    self.log(f"{key} (min): {np.min(vals)}")
                    self.log(f"{key} (max): {np.max(vals)}")
            else:
                self.log(f"{key}: None")
        self.epoch_dict[key] = [] # Clear after logging

    def dump_tabular(self):
        self.log("-" * 30)



class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = list(next_obs)
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_batch_with_history(self, batch_size=32, max_hist_len=100):
        idxs = np.random.randint(max_hist_len, self.size, size=batch_size)
        # History
        if max_hist_len == 0:
            hist_obs = np.zeros([batch_size, 1, self.obs_dim])
            hist_act = np.zeros([batch_size, 1, self.act_dim])
            hist_obs2 = np.zeros([batch_size, 1, self.obs_dim])
            hist_act2 = np.zeros([batch_size, 1, self.act_dim])
            hist_obs_len = np.zeros(batch_size)
            hist_obs2_len = np.zeros(batch_size)
        else:
            hist_obs = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs_len = max_hist_len * np.ones(batch_size)
            hist_obs2 = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act2 = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs2_len = max_hist_len * np.ones(batch_size)

            # Extract history experiences before sampled index
            for i, id in enumerate(idxs):
                hist_start_id = id - max_hist_len
                if hist_start_id < 0:
                    hist_start_id = 0
                if len(np.where(self.done_buf[hist_start_id:id] == 1)[0]) != 0:
                    hist_start_id = hist_start_id + (np.where(self.done_buf[hist_start_id:id] == 1)[0][-1]) + 1
                hist_seg_len = id - hist_start_id
                hist_obs_len[i] = hist_seg_len
                hist_obs[i, :hist_seg_len, :] = self.obs_buf[hist_start_id:id]
                hist_act[i, :hist_seg_len, :] = self.act_buf[hist_start_id:id]
                if hist_seg_len == 0:
                    hist_obs2_len[i] = 1
                else:
                    hist_obs2_len[i] = hist_seg_len
                hist_obs2[i, :hist_seg_len, :] = self.obs2_buf[hist_start_id:id]
                hist_act2[i, :hist_seg_len, :] = self.act_buf[hist_start_id+1:id+1]

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     hist_obs=hist_obs,
                     hist_act=hist_act,
                     hist_obs2=hist_obs2,
                     hist_act2=hist_act2,
                     hist_obs_len=hist_obs_len,
                     hist_obs2_len=hist_obs2_len)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
#######################################################################################

#######################################################################################


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()
        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]

        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size)-1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h+1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim + act_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
                                      nn.Identity()]

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        #    History output mask to reduce disturbance cased by none history memory
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)

        # Current Feature Extraction
        x = torch.cat([obs, act], dim=-1)
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        return torch.squeeze(x, -1), extracted_memory


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()

        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]
        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size) - 1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h + 1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [act_dim]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]), nn.Tanh()]

    def forward(self, obs, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)

        # Current Feature Extraction
        x = obs
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        return self.act_limit * x, extracted_memory


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit=1,
                 critic_mem_pre_lstm_hid_sizes=(128,),
                 critic_mem_lstm_hid_sizes=(128,),
                 critic_mem_after_lstm_hid_size=(128,),
                 critic_cur_feature_hid_sizes=(128,),
                 critic_post_comb_hid_sizes=(128,),
                 critic_hist_with_past_act=False,
                 actor_mem_pre_lstm_hid_sizes=(128,),
                 actor_mem_lstm_hid_sizes=(128,),
                 actor_mem_after_lstm_hid_size=(128,),
                 actor_cur_feature_hid_sizes=(128,),
                 actor_post_comb_hid_sizes=(128,),
                 actor_hist_with_past_act=False):
        super(MLPActorCritic, self).__init__()
        self.q1 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.q2 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.pi = MLPActor(obs_dim, act_dim, act_limit,
                           mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                           mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                           mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                           cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                           post_comb_hid_sizes=actor_post_comb_hid_sizes,
                           hist_with_past_act=actor_hist_with_past_act)

    def act(self, obs, hist_obs=None, hist_act=None, hist_seg_len=None, device=None):
        if (hist_obs is None) or (hist_act is None) or (hist_seg_len is None):
            hist_obs = torch.zeros(1, 1, self.obs_dim).to(device)
            hist_act = torch.zeros(1, 1, self.act_dim).to(device)
            hist_seg_len = torch.zeros(1).to(device)
        with torch.no_grad():
            act, _, = self.pi(obs, hist_obs, hist_act, hist_seg_len)
            return act.cpu().numpy()


#######################################################################################

#######################################################################################
def lstm_td3(resume_exp_dir=None,
             seed=0,
             steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
             polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
             start_steps=10000,
             update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
             noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
             batch_size=100,
             max_hist_len=100,
             use_double_critic = True,
             use_target_policy_smooth = True,
             critic_mem_pre_lstm_hid_sizes=(128,),
             critic_mem_lstm_hid_sizes=(128,),
             critic_mem_after_lstm_hid_size=(128,),
             critic_cur_feature_hid_sizes=(128,),
             critic_post_comb_hid_sizes=(128,),
             critic_hist_with_past_act=False,
             actor_mem_pre_lstm_hid_sizes=(128,),
             actor_mem_lstm_hid_sizes=(128,),
             actor_mem_after_lstm_hid_size=(128,),
             actor_cur_feature_hid_sizes=(128,),
             actor_post_comb_hid_sizes=(128,),
             actor_hist_with_past_act=False,
             logger_kwargs=dict(), save_freq=1):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    """
    # If not going to resume, create new logger.
    if resume_exp_dir is None:
        logger = EpochLogger(**logger_kwargs)
        # logger.save_config(locals()) # Might fail with non-serializable objects
    else:
        logger = EpochLogger(**logger_kwargs)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize HEMSenv
    env = HEMSenv()
    test_env = HEMSenv()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
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
    ac_targ = deepcopy(ac)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ac.to(device)
    ac_targ.to(device)

    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=replay_size)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        h_o, h_a, h_o2, h_a2, h_o_len, h_o2_len = data['hist_obs'], data['hist_act'], data['hist_obs2'], data['hist_act2'], data['hist_obs_len'], data['hist_obs2_len']

        q1, q1_extracted_memory = ac.q1(o, a, h_o, h_a, h_o_len)
        q2, q2_extracted_memory = ac.q2(o, a, h_o, h_a, h_o_len)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ, _ = ac_targ.pi(o2, h_o2, h_a2, h_o2_len)

            # Target policy smoothing
            if use_target_policy_smooth:
                epsilon = torch.randn_like(pi_targ) * target_noise
                epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
                a2 = pi_targ + epsilon
                a2 = torch.clamp(a2, -act_limit, act_limit)
            else:
                a2 = pi_targ

            # Target Q-values
            q1_pi_targ, _ = ac_targ.q1(o2, a2, h_o2, h_a2, h_o2_len)
            q2_pi_targ, _ = ac_targ.q2(o2, a2, h_o2, h_a2, h_o2_len)

            if use_double_critic:
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            else:
                q_pi_targ = q1_pi_targ
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        if use_double_critic:
            loss_q = loss_q1 + loss_q2
        else:
            loss_q = loss_q1

        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy(),
                         Q1ExtractedMemory=q1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
                         Q2ExtractedMemory=q2_extracted_memory.mean(dim=1).detach().cpu().numpy())

        return loss_q, loss_info

    def compute_loss_pi(data):
        o, h_o, h_a, h_o_len = data['obs'], data['hist_obs'], data['hist_act'], data['hist_obs_len']
        a, a_extracted_memory = ac.pi(o, h_o, h_a, h_o_len)
        q1_pi, _ = ac.q1(o, a, h_o, h_a, h_o_len)
        loss_info = dict(ActExtractedMemory=a_extracted_memory.mean(dim=1).detach().cpu().numpy())
        return -q1_pi.mean(), loss_info

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer):
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        if timer % policy_delay == 0:
            for p in q_params:
                p.requires_grad = False

            pi_optimizer.zero_grad()
            loss_pi, loss_info_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            for p in q_params:
                p.requires_grad = True

            logger.store(LossPi=loss_pi.item(), **loss_info_pi)

            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, o_buff, a_buff, o_buff_len, noise_scale, device=None):
        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
        h_l = torch.tensor([o_buff_len]).float().to(device)
        with torch.no_grad():
            a = ac.act(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(device),
                       h_o, h_a, h_l).reshape(act_dim)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0

            if max_hist_len > 0:
                o_buff = np.zeros([max_hist_len, obs_dim])
                a_buff = np.zeros([max_hist_len, act_dim])
                o_buff[0, :] = o
                o_buff_len = 0
            else:
                o_buff = np.zeros([1, obs_dim])
                a_buff = np.zeros([1, act_dim])
                o_buff_len = 0

            while not (d or (ep_len == max_ep_len)):
                a = get_action(o, o_buff, a_buff, o_buff_len, 0, device)
                o2, r, d, _ = test_env.step(a)

                ep_ret += r
                ep_len += 1
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

            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    past_t = 0
    o = env.reset()
    ep_ret, ep_len = 0, 0

    if max_hist_len > 0:
        o_buff = np.zeros([max_hist_len, obs_dim])
        a_buff = np.zeros([max_hist_len, act_dim])
        o_buff[0, :] = o
        o_buff_len = 0
    else:
        o_buff = np.zeros([1, obs_dim])
        a_buff = np.zeros([1, act_dim])
        o_buff_len = 0

    if resume_exp_dir is not None:
        print("Resuming from checkpoint not fully implemented for HEMSenv integration in this demo.")
        
    print("past_t={}".format(past_t))

    for t in range(past_t, total_steps):  # Start from the step after resuming.
        if t > start_steps:
            a = get_action(o, o_buff, a_buff, o_buff_len, act_noise, device)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        logger.log_step_reward(t, r)

        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        replay_buffer.store(o, a, r, o2, d)

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

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

            if max_hist_len > 0:
                o_buff = np.zeros([max_hist_len, obs_dim])
                a_buff = np.zeros([max_hist_len, act_dim])
                o_buff[0, :] = o
                o_buff_len = 0
            else:
                o_buff = np.zeros([1, obs_dim])
                a_buff = np.zeros([1, act_dim])
                o_buff_len = 0

            # Saving checkpoints 
            if (t+1) % steps_per_epoch == 0:
                fpath = 'pyt_save'
                fpath = osp.join(logger.output_dir, fpath)
                os.makedirs(fpath, exist_ok=True)
                model_fname = 'checkpoint-model-' + ('Step-%d' % t if t is not None else '') + '.pt'
                model_elements = {'ac_state_dict': ac.state_dict(),
                                  'target_ac_state_dict': ac_targ.state_dict(),
                                  'pi_optimizer_state_dict': pi_optimizer.state_dict(),
                                  'q_optimizer_state_dict': q_optimizer.state_dict()}
                model_fname = osp.join(fpath, model_fname)
                torch.save(model_elements, model_fname)

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch_with_history(batch_size, max_hist_len)
                batch = {k: v.to(device) for k, v in batch.items()}
                update(data=batch, timer=j)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('Q1ExtractedMemory', with_min_and_max=True)
            logger.log_tabular('Q2ExtractedMemory', with_min_and_max=True)
            logger.log_tabular('ActExtractedMemory', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)

            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()



def str2bool(v):
    """Function used in argument parser for converting string to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def list2tuple(v):
    return tuple(v)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_exp_dir', type=str, default=None, help="The directory of the resuming experiment.")
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--max_hist_len', type=int, default=5)
    parser.add_argument('--use_double_critic', type=str2bool, nargs='?', const=True, default=True,
                        help="Using double critic")
    parser.add_argument('--use_target_policy_smooth', type=str2bool, nargs='?', const=True, default=True,
                        help="Using target policy smoothing")
    parser.add_argument('--critic_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_mem_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--critic_cur_feature_hid_sizes', type=int, nargs="?", default=[128, 128])
    parser.add_argument('--critic_post_comb_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--actor_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_mem_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--actor_cur_feature_hid_sizes', type=int, nargs="?", default=[128, 128])
    parser.add_argument('--actor_post_comb_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--exp_name', type=str, default='lstm_td3_HEMSenv')
    parser.add_argument("--data_dir", type=str, default=str(get_dir("outputs") / 'experiments'))
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_test_episodes', type=int, default=10)

    args = parser.parse_args()

    if args.critic_cur_feature_hid_sizes is None:
        args.critic_cur_feature_hid_sizes = []
    if args.actor_cur_feature_hid_sizes is None:
        args.actor_cur_feature_hid_sizes = []


    if args.resume_exp_dir is None:
        data_dir = osp.join(os.getcwd(), args.data_dir)
        logger_kwargs = dict(output_dir=os.path.join(data_dir, args.exp_name + str(args.seed)), exp_name=args.exp_name, seed=args.seed)
    else:
        logger_kwargs = dict(output_dir=args.resume_exp_dir, exp_name=args.exp_name, seed=args.seed)


    lstm_td3(resume_exp_dir=args.resume_exp_dir,
             gamma=args.gamma, seed=args.seed, epochs=args.epochs,
             steps_per_epoch=args.steps_per_epoch,
             max_hist_len=args.max_hist_len,
             use_double_critic=args.use_double_critic,
             use_target_policy_smooth=args.use_target_policy_smooth,
             critic_mem_pre_lstm_hid_sizes=tuple(args.critic_mem_pre_lstm_hid_sizes),
             critic_mem_lstm_hid_sizes=tuple(args.critic_mem_lstm_hid_sizes),
             critic_mem_after_lstm_hid_size=tuple(args.critic_mem_after_lstm_hid_size),
             critic_cur_feature_hid_sizes=tuple(args.critic_cur_feature_hid_sizes),
             critic_post_comb_hid_sizes=tuple(args.critic_post_comb_hid_sizes),
             actor_mem_pre_lstm_hid_sizes=tuple(args.actor_mem_pre_lstm_hid_sizes),
             actor_mem_lstm_hid_sizes=tuple(args.actor_mem_lstm_hid_sizes),
             actor_mem_after_lstm_hid_size=tuple(args.actor_mem_after_lstm_hid_size),
             actor_cur_feature_hid_sizes=tuple(args.actor_cur_feature_hid_sizes),
             actor_post_comb_hid_sizes=tuple(args.actor_post_comb_hid_sizes),
             actor_hist_with_past_act=args.actor_hist_with_past_act,
             logger_kwargs=logger_kwargs,
             start_steps=args.start_steps,
             update_after=args.update_after,
             update_every=args.update_every,
             batch_size=args.batch_size,
             num_test_episodes=args.num_test_episodes)
