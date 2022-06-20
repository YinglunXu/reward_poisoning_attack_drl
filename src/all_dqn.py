import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import argparse
import json
from utils import redirect_stdout

class ReplayBuffer():
    def __init__(self, max_size, input_shape, num_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class DoubleNetwork(nn.Module):

    def __init__(self, lr, input_dims, num_actions, num_neurons=256):
        super(DoubleNetwork, self).__init__()
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.fc1 = nn.Linear(*self.input_dims, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.V = nn.Linear(num_neurons, 1)
        self.A = nn.Linear(num_neurons, self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        value = self.V(l2)
        advantage = self.A(l2)
        q_value = value + (advantage - advantage.mean())

        return q_value

class DOUBLE(object):

    def __init__(self, env_name, gamma=0.99, epsilon=1.0, lr=3e-3, batch_size=64,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, num_neurons=256, num_test_episodes=10):
        self.name = 'double'
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.input_dims = np.asarray(self.env.reset().shape)
        self.num_actions = self.env.action_space.n
        self.num_neurons = 256
        self.num_test_episodes = num_test_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.lr = lr
        self.action_space = [i for i in range(self.num_actions)]
        self.max_mem_size = max_mem_size
        self.memory = ReplayBuffer(max_mem_size, self.input_dims, self.num_actions)
        self.network = 0

        self.Q_eval_A = DoubleNetwork(self.lr, self.input_dims, self.num_actions, num_neurons)
        self.Q_eval_B = DoubleNetwork(self.lr, self.input_dims, self.num_actions, num_neurons)

        self.Q_target_A = DoubleNetwork(self.lr, self.input_dims, self.num_actions, num_neurons)
        self.Q_target_A.load_state_dict(self.Q_eval_A.state_dict())
        self.Q_target_B = DoubleNetwork(self.lr, self.input_dims, self.num_actions, num_neurons)
        self.Q_target_B.load_state_dict(self.Q_eval_B.state_dict())
        self.Q_target_A.eval()
        self.Q_target_B.eval()

    def store_transitions(self, state, action, reward, state_, done):
        self.memory.store_transitions(state, action, reward, state_, done)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = np.array([state])
            state = torch.tensor(state).to(self.Q_eval_A.device)
            q_value = self.Q_eval_A.forward(state)
            action = torch.argmax(q_value).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def choose_action_test(self, state):
        state = np.array([state])
        state = torch.tensor(state).to(self.Q_eval_A.device)
        q_value = self.Q_eval_A.forward(state)
        action = torch.argmax(q_value).item()
        return action

    def replace_target_network(self):
        self.Q_target_A.load_state_dict(self.Q_eval_A.state_dict())
        self.Q_target_B.load_state_dict(self.Q_eval_B.state_dict())

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.network = np.random.randint(2)
        if self.network == 0:
            self.Q_eval_A.optimizer.zero_grad()
        else:
            self.Q_eval_B.optimizer.zero_grad()
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        state_batch = torch.tensor(states).to(self.Q_eval_A.device)
        new_state_batch = torch.tensor(states_).to(self.Q_eval_A.device)
        reward_batch = torch.tensor(rewards).to(self.Q_eval_A.device)
        terminal_batch = torch.tensor(dones).to(self.Q_eval_A.device)
        action_batch = actions

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        if self.network == 0:
            q_update = self.Q_eval_A.forward(state_batch)
            q_eval = q_update[batch_index, action_batch]
            q_next = self.Q_target_B.forward(new_state_batch)
            q_next[terminal_batch] = 0.0

        else:
            q_update = self.Q_eval_B.forward(state_batch)
            q_eval = q_update[batch_index, action_batch]
            q_next = self.Q_target_A.forward(new_state_batch)
            q_next[terminal_batch] = 0.0

        action_mask = torch.max(q_update, dim=1)[1]
        q_target = reward_batch + self.gamma * q_next[batch_index, action_mask]

        if self.network == 0:
            loss = self.Q_eval_A.loss(q_target, q_eval).to(self.Q_eval_A.device)
            loss.backward()
            self.Q_eval_A.optimizer.step()
        else:
            loss = self.Q_eval_B.loss(q_target, q_eval).to(self.Q_eval_B.device)
            loss.backward()
            self.Q_eval_B.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

    def test(self):
        test_scores = []
        for j in range(self.num_test_episodes):
            state, d, test_score = self.test_env.reset(), False, 0
            state = state.astype('float32')
            while not d:
                state, r, d, _ = self.test_env.step(self.choose_action_test(state))
                state = state.astype('float32')
                test_score += r
            test_scores.append(test_score)
        return test_scores

    def test_target1(self, atk_agent):
        test_scores, sames, differents = [], [], []
        for j in range(self.num_test_episodes):
            state, d, test_score = self.test_env.reset(), False, 0
            state = state.astype('float32')
            same, different = 0, 0
            while not d:
                action = self.choose_action_test(state)
                old_state = state
                state, r, d, _ = self.test_env.step(action)
                state = state.astype('float32')
                if action == atk_agent.choose_action_test(old_state):
                    same += 1
                else:
                    different += 1
                test_score += r
                state_ = state
            test_scores.append(test_score)
            sames.append(same)
            differents.append(different)
        return test_scores, sames, differents

    def reset(self):
        self.memory = ReplayBuffer(self.max_mem_size, self.input_dims, self.num_actions)
        self.Q_eval_A = DoubleNetwork(self.lr, self.input_dims, self.num_actions, self.num_neurons)
        self.Q_eval_B = DoubleNetwork(self.lr, self.input_dims, self.num_actions, self.num_neurons)

        self.Q_target_A = DoubleNetwork(self.lr, self.input_dims, self.num_actions, self.num_neurons)
        self.Q_target_A.load_state_dict(self.Q_eval_A.state_dict())
        self.Q_target_B = DoubleNetwork(self.lr, self.input_dims, self.num_actions, self.num_neurons)
        self.Q_target_B.load_state_dict(self.Q_eval_A.state_dict())
        self.Q_target_A.eval()
        self.Q_target_B.eval()

class DuelingNetwork(nn.Module):

    def __init__(self, lr, input_dims, num_actions, num_neurons=128):
        super(DuelingNetwork, self).__init__()
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.fc1 = nn.Linear(*self.input_dims, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.V = nn.Linear(num_neurons, 1)
        self.A = nn.Linear(num_neurons, self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        value = self.V(l2)
        advantage = self.A(l2)
        q_value = value + (advantage - advantage.mean())

        return q_value

class DUEL(object):
    def __init__(self, env_name, gamma=0.99, epsilon=1.0, lr=3e-3, batch_size=64,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, num_neurons=128, num_test_episodes=10):
        self.name = 'duel'
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.input_dims = np.asarray(self.env.reset().shape)
        self.num_actions = self.env.action_space.n
        self.num_neurons = 128
        self.num_test_episodes = num_test_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.lr = lr
        self.action_space = [i for i in range(self.num_actions)]
        self.max_mem_size = max_mem_size
        self.memory = ReplayBuffer(max_mem_size, self.input_dims, self.num_actions)

        self.Q_eval = DuelingNetwork(self.lr, self.input_dims, self.num_actions, num_neurons)

        self.Q_target = DuelingNetwork(self.lr, self.input_dims, self.num_actions, num_neurons)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()

    def store_transitions(self, state, action, reward, state_, done):
        self.memory.store_transitions(state, action, reward, state_, done)

    def get_Q(self, state):
        state = np.array([state])
        state = torch.tensor(state).to(self.Q_eval.device)
        q_value = self.Q_eval.forward(state)
        return q_value

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            q_value = self.get_Q(state)
            action = torch.argmax(q_value).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def choose_action_test(self, state):
        q_value = self.get_Q(state)
        action = torch.argmax(q_value).item()
        return action

    def replace_target_network(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        state_batch = torch.tensor(states).to(self.Q_eval.device)
        new_state_batch = torch.tensor(states_).to(self.Q_eval.device)
        reward_batch = torch.tensor(rewards).to(self.Q_eval.device)
        terminal_batch = torch.tensor(dones).to(self.Q_eval.device)
        action_batch = actions

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def test(self):
        test_scores = []
        for j in range(self.num_test_episodes):
            state, d, test_score = self.test_env.reset(), False, 0
            state = state.astype('float32')
            while not d:
                state, r, d, _ = self.test_env.step(self.choose_action_test(state))
                state = state.astype('float32')
                test_score += r
            test_scores.append(test_score)
        return test_scores

    def test_target1(self, atk_agent):
        test_scores, sames, differents = [], [], []
        for j in range(self.num_test_episodes):
            state, d, test_score = self.test_env.reset(), False, 0
            state = state.astype('float32')
            same, different = 0, 0
            while not d:
                old_state = state
                state, r, d, _ = self.test_env.step(self.choose_action_test(state))
                state = state.astype('float32')
                if self.choose_action_test(state) == atk_agent.choose_action_test(old_state):
                    same += 1
                else:
                    different += 1
                test_score += r
            test_scores.append(test_score)
            sames.append(same)
            differents.append(different)
        return test_scores, sames, differents
    def reset(self):
        self.memory = ReplayBuffer(self.max_mem_size, self.input_dims, self.num_actions)
        self.Q_eval = DuelingNetwork(self.lr, self.input_dims, self.num_actions, self.num_neurons)

        self.Q_target = DuelingNetwork(self.lr, self.input_dims, self.num_actions, self.num_neurons)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()

def training(agent, dir, atk_agents=[], atk_params=None, n_epochs=100, len_epoch=800, n_runs=1, uniform = False, save_model = False, promote = False):
    target_update = 10
    n_steps = n_epochs * len_epoch
    score_log, c_log = [], []
    for i in range(n_runs):
        print('epoch', i)
        counter, epoch = 0, 0
        test_scores, cur_c = [], []
        C = 0
        if atk_agents != [] or uniform == True:
            C, B1, B2 = atk_params
            C = C*n_steps
        while counter <= n_steps:
            state = agent.env.reset()
            state = state.astype('float32')
            done = False
            score, score_clean = 0, 0
            while done != True:
                action = agent.choose_action(state)
                state_, reward, done, _ = agent.env.step(action)
                state_ = state_.astype('float32')
                score_clean += reward
                if uniform:
                    if C > 0 and score_clean - reward - score <= B1 and np.random.random_sample() < atk_params[0]:
                        reward -= B2
                        C -= 1
                elif atk_agents != None:
                    a_atks = []
                    for atk_agent in atk_agents:
                        a_atk = atk_agent.choose_action_test(state)
                        a_atks.append(a_atk)
                    if C > 0 and abs(score_clean - reward - score) <= B1:
                        for a_atk in a_atks:
                                if a_atk == action:
                                    if not promote:
                                        reward -= B2
                                        C -= 1
                                    else:
                                        reward += B2
                                        C -= 1
                score += reward
                agent.store_transitions(state, action, reward, state_, done)
                agent.learn()
                state = state_
                if counter % target_update == 0:
                    agent.replace_target_network()
                counter += 1
                if counter % len_epoch == 0:
                    test_score = agent.test()
                    test_scores.append(test_score)
                    cur_c.append(C)
                    pfm = sum(test_score)/len(test_score)
                    print(epoch, pfm, C)
                    epoch += 1
                    if save_model:
                        if agent.name == 'double':
                            torch.save(agent.Q_eval_A.state_dict(),
                                       '../models/%s_model/%s_qa_%i' % (agent.name, agent.env_name, pfm))
                            torch.save(agent.Q_eval_B.state_dict(),
                                       '../models/%s_model/%s_qb_%i' % (agent.name, agent.env_name, pfm))
                        if agent.name == 'duel':
                            torch.save(agent.Q_eval.state_dict(),
                                       '../models/%s_model/%s_q_%i' % (agent.name, agent.env_name, pfm))

        score_log.append(test_scores)
        c_log.append(cur_c)
        agent.reset()
        C = 0
        if atk_agents != None or uniform == True:
            C, B1, B2 = atk_params
            C = C*n_steps

    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))

def training_online(agent, dir, atk_agent, phase_change, atk_params=None, n_epochs=100, len_epoch=800, n_runs=1):
    target_update = 10
    n_steps = n_epochs * len_epoch
    score_log, c_log = [], []
    for i in range(n_runs):
        print('epoch', i)
        counter, epoch = 0, 0
        test_scores, cur_c = [], []
        C, B1, B2 = atk_params
        C = C*n_steps
        C_original = C
        phase = 0
        while counter <= n_steps:
            state = agent.env.reset()
            state = state.astype('float32')
            done = False
            score, score_clean, corrupt = 0, 0, 0
            while done != True:
                action = agent.choose_action(state)
                state_, reward, done, _ = agent.env.step(action)
                state_ = state_.astype('float32')
                score_clean += reward
                if phase == 0:
                    if C > 0 and corrupt <= B1:
                        if abs(reward) < B2:
                            corrupt += abs(reward)
                            reward = 0
                            C -= 1
                        elif reward > 0:
                            corrupt += B2
                            reward -= B2
                            C -= 1
                        else:
                            corrupt += B2
                            reward += B2
                            C -= 1
                        # if np.random.random_sample() <= 0.5:
                        #     reward += 0.1
                        # else:
                        #     reward -= 0.1
                    atk_agent.store_transitions(state, action, reward, state_, done)
                    atk_agent.learn()
                elif phase == 1:
                    a_atk = atk_agent.choose_action_test(state)
                    if C > 0 and corrupt <= B1:
                        if a_atk == action:
                            corrupt += B2
                            reward -= B2
                            C -= 1
                        if a_atk != action:
                            corrupt += B2
                            reward += B2
                            C -= 1
                else:
                    a_atk = atk_agent.choose_action_test(state)
                    if C > 0 and corrupt <= B1:
                        if a_atk == action:
                            corrupt += B2
                            reward -= B2
                            C -= 1
                score += reward
                agent.store_transitions(state, action, reward, state_, done)
                agent.learn()
                state = state_
                if counter % target_update == 0:
                    agent.replace_target_network()
                counter += 1
                if counter % len_epoch == 0:
                    test_score = agent.test()
                    test_scores.append(test_score)
                    cur_c.append(C)
                    pfm = sum(test_score)/len(test_score)
                    print(epoch, pfm, C)
                    epoch += 1
                    if C <= phase_change * C_original:
                        phase = 1
                    if C <= 0.5 * phase_change * C_original:
                        phase = 2


        score_log.append(test_scores)
        c_log.append(cur_c)
        agent.reset()
        C, B1, B2 = atk_params
        C = C*n_steps

    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))

def training_target1(agent, dir, atk_agent, atk_params, n_epochs=100, len_epoch=800, n_runs=1):
    target_update = 10
    n_steps = n_epochs * len_epoch
    score_log, c_log, same_log, different_log = [], [], [], []
    for i in range(n_runs):
        print('epoch', i)
        counter, epoch = 0, 0
        test_scores, cur_c, sames, differents = [], [], [], []
        C, B1, B2 = atk_params
        C = C*n_steps
        while counter <= n_steps:
            state = agent.env.reset()
            state = state.astype('float32')
            done = False
            score, score_clean = 0, 0
            while done != True:
                action = agent.choose_action(state)
                state_, reward, done, _ = agent.env.step(action)
                state_ = state_.astype('float32')
                score_clean += reward
                a_atk = atk_agent.choose_action_test(state)
                if a_atk != action:
                    if C > 0 and score_clean - reward - score <= B1:
                        reward -= B2
                        C -= 1
                score += reward
                agent.store_transitions(state, action, reward, state_, done)
                agent.learn()
                state = state_
                if counter % target_update == 0:
                    agent.replace_target_network()
                counter += 1
                if counter % len_epoch == 0:
                    test_score, same, different = agent.test_target1(atk_agent)
                    sames.append(same)
                    differents.append(different)
                    test_scores.append(test_score)
                    cur_c.append(C)
                    pfm = sum(test_score)/len(test_score)
                    same_avg = sum(same)/len(same)
                    different_avg = sum(different)/len(different)
                    print(epoch, pfm, C, same_avg, different_avg)
                    epoch += 1

        score_log.append(test_scores)
        c_log.append(cur_c)
        same_log.append(sames)
        different_log.append(differents)
        agent.reset()
        C, B1, B2 = atk_params
        C = C*n_steps

    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    data['same'] = same_log
    data['different'] = different_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))

def training_target2(agent, dir, atk_agent, atk_params, n_epochs=100, len_epoch=800, n_runs=1):
    target_update = 10
    n_steps = n_epochs * len_epoch
    score_log, c_log, same_log, different_log = [], [], [], []
    for i in range(n_runs):
        print('epoch', i)
        counter, epoch = 0, 0
        test_scores, cur_c, sames, differents = [], [], [], []
        C, B1, B2 = atk_params
        C = C*n_steps
        while counter <= n_steps:
            state = agent.env.reset()
            state = state.astype('float32')
            done = False
            score, score_clean = 0, 0
            while done != True:
                action = agent.choose_action(state)
                state_, reward, done, _ = agent.env.step(action)
                state_ = state_.astype('float32')
                score_clean += reward
                a_atk = atk_agent.choose_action_test(state)
                if a_atk == action:
                    if C > 0 and abs(score_clean - reward - score) <= B1:
                        reward += B2
                        C -= 1
                score += reward
                agent.store_transitions(state, action, reward, state_, done)
                agent.learn()
                state = state_
                if counter % target_update == 0:
                    agent.replace_target_network()
                counter += 1
                if counter % len_epoch == 0:
                    test_score, same, different = agent.test_target1(atk_agent)
                    sames.append(same)
                    differents.append(different)
                    test_scores.append(test_score)
                    cur_c.append(C)
                    pfm = sum(test_score)/len(test_score)
                    same_avg = sum(same)/len(same)
                    different_avg = sum(different)/len(different)
                    print(epoch, pfm, C, same_avg, different_avg)
                    epoch += 1

        score_log.append(test_scores)
        c_log.append(cur_c)
        same_log.append(sames)
        different_log.append(differents)
        agent.reset()
        C, B1, B2 = atk_params
        C = C*n_steps

    data = dict()
    data['C'] = c_log
    data['score'] = score_log
    data['same'] = same_log
    data['different'] = different_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='duel')
    parser.add_argument('--env', type=str, default='Acrobot-v1')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--atk_params", nargs="+", type=float, default=[0.05, 1000, 5])
    parser.add_argument('--atk_alg', type=str, default='duel')
    parser.add_argument("--atk_pfm", nargs="+", type=int, default=None)
    parser.add_argument("--atk_online", action='store_true')
    parser.add_argument("--change", type=float, default=0.5)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=800)
    parser.add_argument("--num_test_episodes", type=int, default=10)
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--uniform", action='store_true')
    parser.add_argument("--target1", action='store_true')
    parser.add_argument("--target2", action='store_true')
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
    print('atk_params:', args.atk_params)
    print('atk_online:', args.atk_online)
    if args.atk_online:
        print('phase_change:', args.change)
    print('steps_per_epoch:', args.steps_per_epoch)
    print('num_test_episodes:', args.num_test_episodes)
    print('uniform_attack:', args.uniform)
    print('target_policy1:', args.target1)
    print('target_policy2:', args.target2)
    print('promote:', args.promote)
    agent = eval(args.alg.upper())(env_name=args.env, num_test_episodes=args.num_test_episodes)
    agent.env.seed(args.seed)
    agent.test_env.seed(args.seed)
    if not args.atk_online:
        if args.target1 == False and args.target2==False:
            atk_agents = []
            if args.atk_alg != None and args.atk_pfm != None:
                for pfm in args.atk_pfm:
                    atk_agent = eval(args.atk_alg.upper())(env_name=args.env, num_test_episodes=args.num_test_episodes)
                    if pfm > -1000:
                        if args.atk_alg == 'duel':
                            atk_agent.Q_eval.load_state_dict(torch.load('../models/%s_model/%s_q_%d' % (args.atk_alg, args.env, pfm)))
                        if args.atk_alg == 'double':
                            atk_agent.Q_eval_A.load_state_dict(
                                torch.load('../models/%s_model/%s_qa_%d' % (args.atk_alg, args.env, pfm)))
                            atk_agent.Q_eval_B.load_state_dict(
                                torch.load('../models/%s_model/%s_qb_%d' % (args.atk_alg, args.env, pfm)))
                    atk_agents.append(atk_agent)
            training(agent = agent, dir = args.dir, atk_agents=atk_agents, atk_params=args.atk_params, len_epoch=args.steps_per_epoch,
                     n_epochs=args.epochs, n_runs=args.n_runs, uniform=args.uniform, save_model=args.save, promote=args.promote)
        elif args.target1 == True:
            atk_agent = eval(args.atk_alg.upper())(env_name=args.env, num_test_episodes=args.num_test_episodes)
            pfm = args.atk_pfm[0]
            if pfm > -1000:
                if args.atk_alg == 'duel':
                    atk_agent.Q_eval.load_state_dict(torch.load('../models/%s_model/%s_q_%d' % (args.atk_alg, args.env, pfm)))
                if args.atk_alg == 'double':
                    atk_agent.Q_eval_A.load_state_dict(
                        torch.load('../models/%s_model/%s_qa_%d' % (args.atk_alg, args.env, pfm)))
                    atk_agent.Q_eval_B.load_state_dict(
                        torch.load('../models/%s_model/%s_qb_%d' % (args.atk_alg, args.env, pfm)))
            training_target1(agent=agent, dir = args.dir, atk_agent=atk_agent, atk_params=args.atk_params, len_epoch=args.steps_per_epoch,
                     n_epochs=args.epochs, n_runs=args.n_runs)
        elif args.target2 == True:
            atk_agent = eval(args.atk_alg.upper())(env_name=args.env, num_test_episodes=args.num_test_episodes)
            pfm = args.atk_pfm[0]
            if pfm > -1000:
                if args.atk_alg == 'duel':
                    atk_agent.Q_eval.load_state_dict(torch.load('../models/%s_model/%s_q_%d' % (args.atk_alg, args.env, pfm)))
                if args.atk_alg == 'double':
                    atk_agent.Q_eval_A.load_state_dict(
                        torch.load('../models/%s_model/%s_qa_%d' % (args.atk_alg, args.env, pfm)))
                    atk_agent.Q_eval_B.load_state_dict(
                        torch.load('../models/%s_model/%s_qb_%d' % (args.atk_alg, args.env, pfm)))
            training_target2(agent=agent, dir = args.dir, atk_agent=atk_agent, atk_params=args.atk_params, len_epoch=args.steps_per_epoch,
                     n_epochs=args.epochs, n_runs=args.n_runs)
    else:
        atk_agent=eval(args.atk_alg.upper())(env_name=args.env, num_test_episodes=args.num_test_episodes)
        training_online(agent, dir = args.dir, atk_agent = atk_agent, phase_change=args.change, atk_params=args.atk_params, n_epochs=args.epochs, len_epoch=args.steps_per_epoch, n_runs=args.n_runs)
