from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
from numpy import linalg as LA
import gym
import time
import argparse
import json
from utils import redirect_stdout
import torch.nn as nn
import itertools
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import scipy.signal
from gym.spaces import Box, Discrete
import os

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
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

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers).to(torch.device('cuda'))

class DDPGActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = (torch.as_tensor(act_limit, dtype=torch.float32)).to(torch.device('cuda'))

    def forward(self, obs):
        return self.act_limit * self.pi(obs)

class DDPGQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class DDPGActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = DDPGActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = DDPGQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.device = torch.device('cuda')

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

class DDPG(object):
    def __init__(self, env_name, ac_kwargs=dict(), replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3,
                 q_lr=1e-3, batch_size=100, act_noise=0.1, num_test_episodes=10, max_ep_len=1000):
        self.name = 'ddpg'
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.act_noise = act_noise
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.ac_kwargs = ac_kwargs
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.ac = DDPGActorCritic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)
        self.pi_lr = pi_lr
        self.q_lr = q_lr

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)
        self.replay_size = replay_size

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o, a, r, o2, d = o.clone().detach().to(self.ac.device), a.clone().detach().to(self.ac.device), r.clone().detach().to(self.ac.device), \
                         o2.clone().detach().to(self.ac.device), d.clone().detach().to(self.ac.device)

        q = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        return loss_q

    def compute_loss_pi(self, data):
        o = data['obs']
        o = o.clone().detach().to(self.ac.device)
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.to(self.ac.device)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.to(self.ac.device)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o):
        noise_scale = self.act_noise
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.ac.device))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def get_action_test(self, o):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.ac.device))
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        ep_rets = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action_test(o))
                ep_ret += r
                ep_len += 1
            ep_rets.append(ep_ret)
        return ep_rets

    def reset(self):
        self.ac = DDPGActorCritic(self.env.observation_space, self.env.action_space, **self.ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=self.q_lr)

class TD3Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = (torch.as_tensor(act_limit, dtype=torch.float32)).to(torch.device('cuda'))

    def forward(self, obs):
        return self.act_limit * self.pi(obs)

class TD3QFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class TD3ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = TD3Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = TD3QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = TD3QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.device = torch.device('cuda')

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

class TD3(object):
    def __init__(self, env_name, ac_kwargs=dict(), replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000):
        self.name = 'td3'
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.ac_kwargs = ac_kwargs
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.ac = TD3ActorCritic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)
        self.replay_size = replay_size

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o, a, r, o2, d = o.clone().detach().to(self.ac.device), a.clone().detach().to(self.ac.device), r.clone().detach().to(self.ac.device), \
                         o2.clone().detach().to(self.ac.device), d.clone().detach().to(self.ac.device)

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    def compute_loss_pi(self, data):
        o = data['obs']
        o = o.clone().detach().to(self.ac.device)
        q1_pi = self.ac.q1(o, self.ac.pi(o))
        return -q1_pi.mean()

    def update(self, data, timer):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.to(self.ac.device)
        loss_q.backward()
        self.q_optimizer.step()

        # Possibly update pi and target networks
        if timer % self.policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.to(self.ac.device)
            loss_pi.backward()
            self.pi_optimizer.step()

            for p in self.q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o):
        noise_scale = self.act_noise
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.ac.device))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def get_action_test(self, o):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.ac.device))
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        ep_rets = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action_test(o))
                ep_ret += r
                ep_len += 1
            ep_rets.append(ep_ret)
        return ep_rets

    def reset(self):
        self.ac = TD3ActorCritic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.q_lr)
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SACActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim).to(torch.device('cuda'))
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim).to(torch.device('cuda'))
        self.act_limit = (torch.as_tensor(act_limit, dtype=torch.float32)).to(torch.device('cuda'))

    def forward(self, obs, deterministic=False, with_logprob=True):
        obs = obs.clone().detach().to(torch.device('cuda'))
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None
        if logp_pi != None:
            logp_pi.to(torch.device('cuda'))
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class SACQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class SACActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SACActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = SACQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = SACQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.device = torch.device('cuda')

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()

class SAC(object):
    def __init__(self, env_name, ac_kwargs=dict(), replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, num_test_episodes=10, max_ep_len=1000):
        self.name = 'sac'
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.lr =lr
        self.alpha = alpha
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.ac_kwargs = ac_kwargs
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.ac = SACActorCritic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)
        self.replay_size = replay_size

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o, a, r, o2, d = o.clone().detach().to(self.ac.device), a.clone().detach().to(self.ac.device), r.clone().detach().to(self.ac.device), \
                         o2.clone().detach().to(self.ac.device), d.clone().detach().to(self.ac.device)
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        o = o.clone().detach().to(self.ac.device)
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        return loss_pi

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.to(self.ac.device)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.to(self.ac.device)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), False)

    def get_action_test(self, o):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), True)

    def test_agent(self):
        ep_rets = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action_test(o))
                ep_ret += r
                ep_len += 1
            ep_rets.append(ep_ret)
        return ep_rets

    def reset(self):
        self.ac = SACActorCritic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class PPOCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class PPOGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std)).to(torch.device('cuda'))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std).to(torch.device('cuda'))
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

class PPOCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.

class PPOActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = PPOGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = PPOCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = PPOCritic(obs_dim, hidden_sizes, activation)
        self.device = torch.device('cuda')

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

class PPO(object):
    def __init__(self, env_name, ac_kwargs=dict(), seed=0, steps_per_epoch=4000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
        target_kl=0.01, num_test_episodes = 10, max_ep_len = 1000):

        setup_pytorch_for_mpi()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.name = 'ppo'
        self.ac_kwargs = ac_kwargs
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)

        # Instantiate environment
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.act_limit = self.env.action_space.high[0]
        # Create actor-critic module
        self.ac = PPOActorCritic(self.env.observation_space, self.env.action_space, **ac_kwargs)

        # Sync params across processes
        sync_params(self.ac)

        # Set up experience buffer
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, gamma, lam)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs, act, adv, logp_old = obs.clone().detach().to(self.ac.device), act.clone().detach().to(self.ac.device), \
                                  adv.clone().detach().to(self.ac.device), logp_old.clone().detach().to(self.ac.device)
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        obs, ret = obs.clone().detach().to(self.ac.device), ret.clone().detach().to(self.ac.device)
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self):
        data = self.buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.to(self.ac.device)
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
            self.pi_optimizer.step()


        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()

    def reset(self):
        self.ac = PPOActorCritic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        sync_params(self.ac)
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, self.gamma, self.lam)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

    def test(self):
        ep_rets = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action_test(o))
                ep_ret += r
                ep_len += 1
            ep_rets.append(ep_ret)
        return ep_rets

    def get_action(self, obs):
        return self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.ac.device))[0]

    def get_action_test(self, obs):
        return self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.ac.device))[0]

def training(agent, dir, atk_agents=[], atk_params=None, atk_radius = 2,
         steps_per_epoch=4000, epochs=100, batch_size=100, start_steps=10000,
         update_after=1000, update_every=50, max_ep_len=1000, save_model=False, n_runs=1, is_uniform = False, atk_start = 0, promote = False):

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    C, B1, B2 = 0, 0, 0
    if atk_agents != [] or is_uniform:
        [C, B1, B2] = atk_params
        C = int(C * total_steps)

    score_log, c_log = [], []
    for i in range(n_runs):
        print('episode', i)
        # Main loop: collect experience in env and update/log each epoch
        score, cur_C = [], []
        o, ep_ret, ep_len, ep_ret_clean = agent.env.reset(), 0, 0, 0
        for t in range(total_steps):
            if t > start_steps:
                a = agent.get_action(o)
            else:
                a = agent.env.action_space.sample()

            # Step the env
            o2, r, d, _ = agent.env.step(a)

            ep_ret_clean += r
            if is_uniform:
                if t >= atk_start and C > 0 and ep_ret_clean - r - ep_ret <= B1 and np.random.random_sample() < atk_params[0]:
                    r -= B2
                    C -= 1

            elif atk_agents != []:
                a_atks = []
                for atk_agent in atk_agents:
                    a_atk = atk_agent.get_action_test(o)
                    if atk_agent.name == 'ppo':
                        a_atk = np.clip(a_atk, -agent.act_limit, agent.act_limit)
                    a_atks.append(a_atk)
                if t>= atk_start and C > 0 and abs(ep_ret_clean - r - ep_ret) <= B1:
                    if not promote:
                        for a_atk in a_atks:
                            if LA.norm(a - a_atk) <= atk_radius * agent.act_limit:
                                r -= B2
                                C -= 1
                                break
                    else:
                        for a_atk in a_atks:
                            if LA.norm(a - a_atk) <= atk_radius * agent.act_limit:
                                r += B2
                                C -= 1
                                break

            ep_ret += r
            ep_len += 1

            d = False if ep_len == max_ep_len else d

            agent.replay_buffer.store(o, a, r, o2, d)

            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                o, ep_ret, ep_len, ep_ret_clean = agent.env.reset(), 0, 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = agent.replay_buffer.sample_batch(batch_size)
                    if agent.name == 'td3':
                        agent.update(data=batch, timer = j)
                    elif agent.name == 'ddpg' or 'sac':
                        agent.update(data=batch)

            # End of epoch handling
            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                # Test the performance of the deterministic version of the agent.

                ep_rets = agent.test_agent()
                score.append(ep_rets)
                pfm = int(sum(ep_rets) / len(ep_rets))
                if save_model:
                    if agent.name == 'ddpg':
                        torch.save(agent.ac.pi.state_dict(), '../models/%s_model/%s_pi_%i' % (agent.name, agent.env_name, pfm))
                        torch.save(agent.ac.q.state_dict(), '../models/%s_model/%s_q_%i' % (agent.name, agent.env_name, pfm))
                    else:
                        torch.save(agent.ac.pi.state_dict(), '../models/%s_model/%s_pi_%i' % (agent.name, agent.env_name, pfm))
                        torch.save(agent.ac.q1.state_dict(), '../models/%s_model/%s_q1_%i' % (agent.name, agent.env_name, pfm))
                        torch.save(agent.ac.q2.state_dict(), '../models/%s_model/%s_q2_%i' % (agent.name, agent.env_name, pfm))
                cur_C.append(C)
                print(epoch, pfm, C)
        c_log.append(cur_C)
        score_log.append(score)
        agent.reset()
        C = 0
        if atk_agents != [] or is_uniform:
            [C, B1, B2] = atk_params
            C = int(C * total_steps)
    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    with open(os.path.join(dir, 'outputs.json'), 'w') as f:
        f.write(json.dumps(data))

def training_online_atk(agent, atk_agent, dir, change, atk_params=None, atk_radius = 2,
         steps_per_epoch=4000, epochs=100, batch_size=100, start_steps=10000,
         update_after=1000, update_every=50,max_ep_len=1000, n_runs=1, phase_change = 0.8):

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    [C, B1, B2] = atk_params
    C = int(C * total_steps)
    C_original = C
    score_log, c_log = [], []
    for i in range(n_runs):
        print('episode', i)
        # Main loop: collect experience in env and update/log each epoch
        score, cur_C = [], []
        o, ep_ret, ep_len, ep_ret_clean, corrupt = agent.env.reset(), 0, 0, 0, 0
        phase = 0
        for t in range(total_steps):
            if t > start_steps:
                a = agent.get_action(o)
            else:
                a = agent.env.action_space.sample()

            # Step the env
            o2, r, d, _ = agent.env.step(a)
            ep_ret_clean += r
            if phase == 0:
                atk_agent.replay_buffer.store(o, a, r, o2, d)
                if C > 0 and corrupt <= B1:
                    if abs(r) < B2:
                        corrupt += abs(r)
                        r = 0
                        C -= 1
                    elif r > 0:
                        corrupt += B2
                        r -= B2
                        C -= 1
                    else:
                        corrupt += B2
                        r += B2
                        C -= 1
            elif phase == 1:
                a_atk = atk_agent.get_action_test(o)
                if C > 0 and ep_ret_clean - r -ep_ret <= B1 and corrupt <= B1:
                    if LA.norm(a-a_atk) <= atk_radius*agent.act_limit:
                        r -= B2
                        corrupt += B2
                        C -= 1
                    else:
                        r += B2
                        corrupt += B2
                        C -= 1
            if phase == 2:
                a_atk = atk_agent.get_action_test(o)
                if C > 0 and ep_ret_clean - r -ep_ret <= B1 and corrupt <= B1:
                    if LA.norm(a-a_atk) <= atk_radius*agent.act_limit:
                        r -= B2
                        corrupt += B2
                        C -= 1
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            # Store experience to replay buffer
            agent.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                o, ep_ret, ep_len, ep_ret_clean, corrupt = agent.env.reset(), 0, 0, 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = agent.replay_buffer.sample_batch(batch_size)
                    if agent.name == 'td3':
                        agent.update(data=batch, timer = j)
                    elif agent.name == 'ddpg' or 'sac':
                        agent.update(data=batch)
                if phase == 0:
                    for j in range(update_every):
                        batch = atk_agent.replay_buffer.sample_batch(batch_size)
                        if atk_agent.name == 'td3':
                            atk_agent.update(data=batch, timer=j)
                        elif atk_agent.name == 'ddpg' or 'sac':
                            atk_agent.update(data=batch)

            # End of epoch handling
            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                # Test the performance of the deterministic version of the agent.

                ep_rets = agent.test_agent()
                score.append(ep_rets)
                pfm = int(sum(ep_rets) / len(ep_rets))
                print(epoch, pfm, C)
                cur_C.append(C)
                if C <= phase_change * C_original:
                    phase = 1
                if C <= 0.5 * phase_change * C_original:
                    phase = 2

        c_log.append(cur_C)
        score_log.append(score)
        agent.reset()
        atk_agent.reset()
        C = int(atk_params[0] * total_steps)
    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))

def ppo_training_online(agent, dir, atk_agent, change, atk_params=None, atk_radius = 2,
         steps_per_epoch=4000, epochs=200, n_runs=1, max_ep_len = 1000):

    [C, B1, B2] = atk_params
    C = int(C * epochs * steps_per_epoch)
    score_log, c_log = [], []
    for i in range(n_runs):
        print('episode', i)
        score, cur_C = [], []
        o, ep_ret, ep_ret_clean, ep_len = agent.env.reset(), 0, 0, 0
        random_agent = eval(atk_agent.name.upper())(env_name=agent.env_name, ac_kwargs=agent.ac_kwargs)
        phase = 0
        for epoch in range(epochs):
            for t in range(steps_per_epoch):
                a, v, logp = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(agent.ac.device))
                a = np.clip(a, -agent.act_limit, agent.act_limit)
                next_o, r, d, _ = agent.env.step(a)
                ep_ret_clean += r
                if phase == 0:
                    atk_agent.replay_buffer.store(o, a, r, next_o, d)
                    a_atk = random_agent.get_action_test(o)
                else:
                    a_atk = atk_agent.get_action_test(o)
                if C > 0 and ep_ret_clean - r - ep_ret <= B1 and LA.norm(a - a_atk) <= atk_radius * agent.act_limit:
                    r -= B2
                    C -= 1

                ep_ret += r
                ep_len += 1

                # save and log
                agent.buf.store(o, a, r, v, logp)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(agent.ac.device))
                    else:
                        v = 0
                    agent.buf.finish_path(v)
                    o, ep_ret, ep_ret_clean, ep_len = agent.env.reset(), 0, 0, 0

            # Perform PPO update!
            agent.update()
            ep_rets = agent.test()
            pfm = int(sum(ep_rets) / len(ep_rets))
            score.append(ep_rets)
            cur_C.append(C)
            if epoch >= change:
                phase = 1
            print(epoch, pfm, C)
        c_log.append(cur_C)
        score_log.append(score)
        agent.reset()
        atk_agent.reset()
        [C, B1, B2] = atk_params
        C = int(C * epochs * steps_per_epoch)
    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))


def ppo_training(agent, dir, atk_agents=[], atk_params=None, atk_radius = 2,
         steps_per_epoch=4000, epochs=200, save_model=False, n_runs=1, max_ep_len = 1000, is_uniform=False, promote = False):
    C, B1, B2 = 0, 0, 0
    if atk_agents != [] or is_uniform:
        [C, B1, B2] = atk_params
        C = int(C * epochs * steps_per_epoch)
    score_log, c_log = [], []
    for i in range(n_runs):
        print('episode', i)
        score, cur_C = [], []
        o, ep_ret, ep_ret_clean, ep_len = agent.env.reset(), 0, 0, 0
        for epoch in range(epochs):
            for t in range(steps_per_epoch):
                a, v, logp = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(agent.ac.device))
                # a = np.clip(a, -agent.act_limit, agent.act_limit)
                next_o, r, d, _ = agent.env.step(a)
                ep_ret_clean += r

                if is_uniform:
                    if C > 0 and ep_ret_clean - r - ep_ret <= B1 and np.random.random_sample() < atk_params[0]:
                        r -= B2
                        C -= 1

                elif atk_agents != []:
                    a_atks = []
                    for atk_agent in atk_agents:
                        a_atk = atk_agent.get_action_test(o)
                        a_atks.append(a_atk)
                    if C > 0 and abs(ep_ret_clean - r - ep_ret) <= B1:
                        for a_atk in a_atks:
                            if LA.norm(a - a_atk) <= atk_radius * agent.act_limit:
                                if not promote:
                                    r -= B2
                                    C -= 1
                                    break
                                else:
                                    r += B2
                                    C -= 1
                                    break
                ep_ret += r
                ep_len += 1

                # save and log
                agent.buf.store(o, a, r, v, logp)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(agent.ac.device))
                    else:
                        v = 0
                    agent.buf.finish_path(v)
                    o, ep_ret, ep_ret_clean, ep_len = agent.env.reset(), 0, 0, 0

            # Perform PPO update!
            agent.update()
            ep_rets = agent.test()
            pfm = int(sum(ep_rets) / len(ep_rets))
            if save_model:
                torch.save(agent.ac.pi.state_dict(), '../models/%s_model/%s_pi_%i' % (agent.name, agent.env_name, pfm))
                torch.save(agent.ac.v.state_dict(), '../models/%s_model/%s_v_%i' % (agent.name, agent.env_name, pfm))
            score.append(ep_rets)
            cur_C.append(C)
            print(epoch, pfm, C)
        c_log.append(cur_C)
        score_log.append(score)
        agent.reset()
        if atk_agents != [] or is_uniform:
            [C, B1, B2] = atk_params
            C = int(C * epochs * steps_per_epoch)
    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))

def training_target_atk(agent, dir, atk_agents, atk_params, sim_radius = 0.2,
         steps_per_epoch=4000, epochs=100, batch_size=100, start_steps=10000,
         update_after=1000, update_every=50,max_ep_len=1000, n_runs=1):

    total_steps = steps_per_epoch * epochs

    [C, B1, B2] = atk_params
    C = C * total_steps

    score_log, c_log = [], []
    for i in range(n_runs):
        print('episode', i)
        score, cur_C = [], []
        o, ep_ret, ep_len, ep_ret_clean = agent.env.reset(), 0, 0, 0
        for t in range(total_steps):
            if t > start_steps:
                a = agent.get_action(o)
            else:
                a = agent.env.action_space.sample()

            o2, r, d, _ = agent.env.step(a)

            ep_ret_clean += r

            a_atks = []
            for atk_agent in atk_agents:
                a_atk = atk_agent.get_action_test(o)
                # if atk_agent.name == 'ppo':
                #     a_atk = np.clip(a_atk, -agent.act_limit, agent.act_limit)
                a_atks.append(a_atk)
            if C > 0 and ep_ret_clean - ep_ret <= B1:
                for a_atk in a_atks:
                    if LA.norm(a - a_atk) >= sim_radius * agent.act_limit:
                        r -= B2
                        C -= 1
                        break

            ep_ret += r
            ep_len += 1

            d = False if ep_len == max_ep_len else d

            agent.replay_buffer.store(o, a, r, o2, d)

            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                o, ep_ret, ep_len, ep_ret_clean = agent.env.reset(), 0, 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = agent.replay_buffer.sample_batch(batch_size)
                    if agent.name == 'td3':
                        agent.update(data=batch, timer = j)
                    elif agent.name == 'ddpg' or 'sac':
                        agent.update(data=batch)

            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                # Test the performance of the deterministic version of the agent.

                ep_rets = agent.test_agent()
                score.append(ep_rets)
                pfm = int(sum(ep_rets) / len(ep_rets))
                cur_C.append(C)
                print(epoch, pfm, C)



        c_log.append(cur_C)
        score_log.append(score)
        agent.reset()
        [C, B1, B2] = atk_params
        C = C * total_steps
    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))


def ppo_training_target(agent, dir, atk_agent, atk_params, sim_radius = 2,
         steps_per_epoch=4000, epochs=500, n_runs=1, max_ep_len = 1000):

    [C, B1, B2] = atk_params
    C = int(C * epochs * steps_per_epoch)

    score_log, c_log = [], []
    for i in range(n_runs):
        print('episode', i)
        score, cur_C = [], []
        o, ep_ret, ep_ret_clean, ep_len = agent.env.reset(), 0, 0, 0
        for epoch in range(epochs):
            for t in range(steps_per_epoch):
                a, v, logp = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(agent.ac.device))

                next_o, r, d, _ = agent.env.step(a)

                ep_ret_clean += r


                a_atk = atk_agent.get_action_test(o)
                if C > 0 and ep_ret_clean - r - ep_ret <= B1:
                    if LA.norm(a - a_atk) >= sim_radius * agent.act_limit:
                        r -= B2
                        C -= 1
                ep_ret += r
                ep_len += 1

                # save and log
                agent.buf.store(o, a, r, v, logp)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(agent.ac.device))
                    else:
                        v = 0
                    agent.buf.finish_path(v)
                    o, ep_ret, ep_len = agent.env.reset(), 0, 0

            # Perform PPO update!
            agent.update()
            ep_rets = agent.test()
            pfm = int(sum(ep_rets) / len(ep_rets))
            score.append(ep_rets)
            cur_C.append(C)
            print(epoch, pfm, C)
        c_log.append(cur_C)
        score_log.append(score)
        agent.reset()
        [C, B1, B2] = atk_params
        C = int(C * epochs * steps_per_epoch)
    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='ddpg')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--atk_radius', type=float, default=2)
    parser.add_argument("--atk_params", nargs="+", type=float, default=[])
    parser.add_argument('--atk_alg', type=str, default=None)
    parser.add_argument("--atk_pfm", nargs="+", type=int, default=None)
    parser.add_argument("--atk_online", action='store_true')
    parser.add_argument("--change", type=float, default=0.8)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--num_test_episodes", type=int, default=10)
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--uniform", action='store_true')
    parser.add_argument("--target", action='store_true')
    parser.add_argument("--atk_start", type=int, default=0)
    parser.add_argument("--promote", action='store_true')
    parser.add_argument("--dir", type=str, default='../tmp')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    redirect_stdout(open(os.path.join(args.dir, 'outputs.txt'), 'w'))

    print('seed:', args.seed)
    print('alg:', args.alg)
    print('env_name:', args.env)
    print('atk_alg:', args.atk_alg)
    print('atk_pfm:', args.atk_pfm)
    print('atk_radius:', args.atk_radius)
    print('atk_params:', args.atk_params)
    print('atk_start:', args.atk_start)
    print('atk_online:', args.atk_online)
    if args.atk_online:
        print('change:', args.change)
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    print('model_structure:', ac_kwargs)
    print('steps_per_epoch:', args.steps_per_epoch)
    print('num_test_episodes:', args.num_test_episodes)
    print('uniform_attack:', args.uniform)
    print('target_policy:', args.target)
    print('promote:', args.promote)
    agent = eval(args.alg.upper())(env_name=args.env, ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                                   gamma=args.gamma, num_test_episodes=args.num_test_episodes)
    agent.env.seed(args.seed)
    agent.test_env.seed(args.seed)
    if not args.target:
        if not args.atk_online:
            atk_agents = []
            if args.atk_alg != None and args.atk_pfm != None:
                for pfm in args.atk_pfm:
                    atk_agent = eval(args.atk_alg.upper())(env_name=args.env, ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma)
                    if pfm > -1000:
                        if args.atk_alg == 'ddpg':
                            atk_agent.ac.pi.load_state_dict(torch.load('../models/%s_model/%s_pi_%d' % (args.atk_alg, args.env, pfm)))
                            atk_agent.ac.q.load_state_dict(torch.load('../models/%s_model/%s_q_%d' % (args.atk_alg, args.env, pfm)))
                        elif args.atk_alg == 'td3' or args.atk_alg == 'sac':
                            atk_agent.ac.pi.load_state_dict(torch.load('../models/%s_model/%s_pi_%d' % (args.atk_alg, args.env, pfm)))
                            atk_agent.ac.q1.load_state_dict(torch.load('../models/%s_model/%s_q1_%d' % (args.atk_alg, args.env, pfm)))
                            atk_agent.ac.q2.load_state_dict(torch.load('../models/%s_model/%s_q2_%d' % (args.atk_alg, args.env, pfm)))
                        elif args.atk_alg == 'ppo':
                            atk_agent.ac.pi.load_state_dict(torch.load('../models/%s_model/%s_pi_%d' % (args.atk_alg, args.env, pfm)))
                            atk_agent.ac.v.load_state_dict(torch.load('../models/%s_model/%s_v_%d' % (args.atk_alg, args.env, pfm)))
                    atk_agents.append(atk_agent)
            if args.alg == 'ppo':
                ppo_training(agent = agent, dir = args.dir, atk_agents=atk_agents, atk_params=args.atk_params, atk_radius = args.atk_radius, steps_per_epoch=args.steps_per_epoch,
                         epochs=args.epochs, save_model=args.save, n_runs=args.n_runs, is_uniform=args.uniform, promote=args.promote)
            else:
                training(agent = agent, dir = args.dir, atk_agents=atk_agents, atk_params=args.atk_params, atk_radius = args.atk_radius, steps_per_epoch=args.steps_per_epoch,
                         epochs=args.epochs, save_model=args.save, n_runs=args.n_runs, is_uniform=args.uniform, atk_start=args.atk_start, promote=args.promote)
        else:
            atk_agent = eval(args.atk_alg.upper())(env_name=args.env, ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                                                   gamma=args.gamma)
            training_online_atk(agent = agent, dir = args.dir, atk_agent = atk_agent, change=args.change, atk_params=args.atk_params, atk_radius=args.atk_radius,
                                epochs=args.epochs, n_runs=args.n_runs, phase_change=args.change)
    else:
        atk_agents = []
        for pfm in args.atk_pfm:
            atk_agent = eval(args.atk_alg.upper())(env_name=args.env, ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                                                   gamma=args.gamma)
            if pfm > -1000:
                if args.atk_alg == 'ddpg':
                    atk_agent.ac.pi.load_state_dict(
                        torch.load('../models/%s_model/%s_pi_%d' % (args.atk_alg, args.env, pfm)))
                    atk_agent.ac.q.load_state_dict(
                        torch.load('../models/%s_model/%s_q_%d' % (args.atk_alg, args.env, pfm)))
                elif args.atk_alg == 'td3' or 'sac':
                    atk_agent.ac.pi.load_state_dict(
                        torch.load('../models/%s_model/%s_pi_%d' % (args.atk_alg, args.env, pfm)))
                    atk_agent.ac.q1.load_state_dict(
                        torch.load('../models/%s_model/%s_q1_%d' % (args.atk_alg, args.env, pfm)))
                    atk_agent.ac.q2.load_state_dict(
                        torch.load('../models/%s_model/%s_q2_%d' % (args.atk_alg, args.env, pfm)))
                elif args.atk_alg == 'ppo':
                    atk_agent.ac.pi.load_state_dict(
                        torch.load('../models/%s_model/%s_pi_%d' % (args.atk_alg, args.env, pfm)))
                    atk_agent.ac.v.load_state_dict(
                        torch.load('../models/%s_model/%s_v_%d' % (args.atk_alg, args.env, pfm)))
            atk_agents.append(atk_agent)
        if args.alg != 'ppo':
            training_target_atk(agent=agent, dir = args.dir, atk_agents=atk_agents, atk_params=args.atk_params, sim_radius = args.atk_radius, steps_per_epoch=args.steps_per_epoch,
                             epochs=args.epochs, n_runs=args.n_runs)
        else:
            ppo_training_target(agent=agent, dir = args.dir, atk_agent=atk_agents[0], atk_params=args.atk_params,
                                sim_radius=args.atk_radius, steps_per_epoch=args.steps_per_epoch,
                                epochs=args.epochs, n_runs=args.n_runs)