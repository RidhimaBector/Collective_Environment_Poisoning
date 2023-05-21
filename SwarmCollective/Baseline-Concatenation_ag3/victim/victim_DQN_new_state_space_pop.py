import math
import random
import numpy as np
import copy

from collections import namedtuple
from collections import defaultdict
from itertools import count
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 

from utils import utils_buf, utils_attack
from envs.target_def import TARGET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
#T_max = 100 #config.VICTIM.TMAX

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

""" QNetwork """
class DQN(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=32, fc2_units=16, learning_rate=0.001, init_w=3e-3):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.init_weights(init_w)

        #self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        #self.loss = nn.MSELoss()
        #self.to(DEVICE) #?????

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

""" Replay Buffer """
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class VictimAgent_DQN_Pop():
    
    def __init__(self, env, MEMORY_SIZE, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, EPSILON, EPSILON_DECAY, TARGET_UPDATE, n_states, n_actions, learning_rate=0.001, no_of_agents=1):
        self.env = env
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.EPSILON = EPSILON
        self.EPSILON_DECAY = EPSILON_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.learning_rate = learning_rate
        
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.policy_net = DQN(n_states, n_actions).to(device)
        self.target_net = DQN(n_states, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=learning_rate)
        self.replay_buffer = ReplayMemory(1000000)
        
        #self.steps_done = 0
        self.no_of_agents = no_of_agents
        self.MEM_array, self.approxQ_data_array, self.LA_array, self.approxQ_array = [], [], [], []
        for k in range(self.no_of_agents):
            self.MEM_array.append(utils_buf.Memory(self.MEMORY_SIZE))
            self.approxQ_data_array.append([[-1]*10 for _ in range(16)]) #[[-1]*10]*16)
            self.LA_array.append(np.ones((16,2))*-1)
            self.LA_array[-1][:,0] = np.arange(16)
            self.approxQ_array.append(np.zeros((16,4)))

        
    def reset(self):
        self.policy_net = DQN(self.n_states, self.n_actions).to(device)
        self.target_net = DQN(self.n_states, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=self.learning_rate)
        self.replay_buffer = ReplayMemory(1000000)
        
        #self.steps_done = 0        
        self.MEM_array, self.approxQ_data_array, self.LA_array, self.approxQ_array = [], [], [], []
        for k in range(self.no_of_agents):
            self.MEM_array.append(utils_buf.Memory(self.MEMORY_SIZE))
            self.approxQ_data_array.append([[-1]*10 for _ in range(16)]) #[[-1]*10]*16) 
            self.LA_array.append(np.ones((16,2))*-1)
            self.LA_array[-1][:,0] = np.arange(16)
            self.approxQ_array.append(np.zeros((16,4)))
    
    
    def compute_approxQ(self):
        
        for k in range(self.no_of_agents):
            
            for state in range(self.env.nS):
                
                for action in range(self.env.nA):
                
                    self.approxQ_array[k][state][action] = self.approxQ_data_array[k][state].count(action)
        
        

    def select_action(self, state):
        #global steps_done
        sample = random.random()
        #eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        #self.steps_done += 1
        eps_threshold = self.EPSILON
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return
        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. 
        state_action_values = self.policy_net(state_batch.float()).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.float()).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        
    def train_model(self, num_episodes, t_max):
        self.EPSILON = 0.1
        
        for i_episode in range(num_episodes):
            # log
            score_array = [0]*self.no_of_agents
            trajectory_list_array = [[]]*self.no_of_agents
            
            # reset env and victim
            state_array, state_complete_array = self.env.reset()
            for k in range(self.no_of_agents):
                state_complete_array[k] = torch.from_numpy(state_complete_array[k]).view(1,self.n_states).to(device)
            
            for t in range(t_max):
                
                action_array = []
                for k in range(self.no_of_agents):
                    action = self.select_action(state_complete_array[k]).to(device)
                    action_array.append(action)
                    
                    self.approxQ_data_array[k][state_array[k]].pop(0)
                    self.approxQ_data_array[k][state_array[k]].append(action.item())
                    self.LA_array[k][state_array[k], 1] = action.item()
                    
                    # store the transition(s,a)
                    t_sample = []
                    if t != 0:
                        t_sample.append(state_array[k])
                        t_sample.append(action.item())
                        trajectory_list_array[k].append(t_sample)
                
                self.EPSILON = self.EPSILON * self.EPSILON_DECAY
                
#               self.env.render()
                next_state_array, next_state_complete_array, reward_array, done_array, _ = self.env.step(action_array)
                
                for k in range(self.no_of_agents):
                    next_state_complete_array[k] = torch.from_numpy(next_state_complete_array[k]).view(1,self.n_states).to(device)
                    reward_array[k] = torch.tensor([reward_array[k]], dtype=torch.float32, device=device)
                    # Store the transition (s,a,s',r) in memory
                    self.replay_buffer.push(state_complete_array[k], action_array[k], next_state_complete_array[k], reward_array[k])

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                
                # Move to the next state
                state_complete_array = copy.deepcopy(next_state_complete_array)
                state_array = copy.deepcopy(next_state_array)
                for k in range(self.no_of_agents):
                    score_array[k] += reward_array[k].item()

                if ((sum(done_array) > 0) or (t+1 == t_max)):
                    # print("DQN-LunarLander: episode = {} | t = {} | score = {}".format(i_episode, t, round(score.item(),0)))
                    break
                    
            # Trajectory to MEMORY with head_padding
            for k in range(self.no_of_agents):
                
                if len(trajectory_list_array[k]) < LEN_TRAJECTORY:
                    padding_state = 0 #torch.zeros(state.shape, device=device)
                    padding_action = 0 #torch.zeros(action.shape, device=device)
                    n_padding = LEN_TRAJECTORY - len(trajectory_list_array[k])
                    for i in range(n_padding):
                        self.MEM_array[k].push(padding_state, padding_action)
                    for i in range(len(trajectory_list_array[k])):
                        state = trajectory_list_array[k][i][0]
                        action = trajectory_list_array[k][i][1]
                        self.MEM_array[k].push(state, action)
                else:
                    for i in range(LEN_TRAJECTORY):
                        state = trajectory_list_array[k][i][0]
                        action = trajectory_list_array[k][i][1]
                        self.MEM_array[k].push(state, action)

            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.compute_approxQ()


    def eval_model(self):
        score_array = [0]*self.no_of_agents
        T_max = 100 #2000
        
        # reset env and victim
        state_array, state_complete_array = self.env.reset()
        for k in range(self.no_of_agents):
            state_complete_array[k] = torch.from_numpy(state_complete_array[k]).view(1,self.n_states).to(device)
                
        for t in range(T_max):
            
            action_array = []
            for k in range(self.no_of_agents):
                action = self.policy_net(state_complete_array[k]).max(1)[1].view(1, 1).to(device)
                action_array.append(action)
                
                self.approxQ_data_array[k][state_array[k]].pop(0)
                self.approxQ_data_array[k][state_array[k]].append(action.item())
                self.LA_array[k][state_array[k], 1] = action.item()

            next_state_array, next_state_complete_array, reward_array, done_array, _ = self.env.step(action_array)
                
            for k in range(self.no_of_agents):
                next_state_complete_array[k] = torch.from_numpy(next_state_complete_array[k]).view(1,self.n_states).to(device)
                reward_array[k] = torch.tensor([reward_array[k]], dtype=torch.float32, device=device)

            state_complete_array = copy.deepcopy(next_state_complete_array)
            state_array = copy.deepcopy(next_state_array)
            for k in range(self.no_of_agents):
                score_array[k] += reward_array[k].item()

            if ((sum(done_array) > 0) or (t+1 == t_max)):
                # print("DQN-LunarLander: episode = {} | t = {} | score = {}".format(i_episode, t, round(score.item(),0)))
                break
        
        return score_array
        
        
    def train_for_eval(self, num_episodes, t_max):
        self.EPSILON = 0.1
        accuracy_episode_array = [[] for _ in range(self.no_of_agents)]
        accuracy_softmax_episode_array = [[] for _ in range(self.no_of_agents)]
        accuracy_softmax_complete_episode_array = [[] for _ in range(self.no_of_agents)]
        
        for i_episode in range(num_episodes):
            # log
            score_array = [0]*self.no_of_agents
            trajectory_list_array = [[]]*self.no_of_agents
            
            # reset env and victim
            state_array, state_complete_array = self.env.reset()
            for k in range(self.no_of_agents):
                state_complete_array[k] = torch.from_numpy(state_complete_array[k]).view(1,self.n_states).to(device)
            
            for t in range(t_max):
                
                action_array = []
                for k in range(self.no_of_agents):
                    action = self.select_action(state_complete_array[k]).to(device)
                    action_array.append(action)
                    
                    self.approxQ_data_array[k][state_array[k]].pop(0)
                    self.approxQ_data_array[k][state_array[k]].append(action.item())
                    self.LA_array[k][state_array[k], 1] = action.item()
                    
                    # store the transition(s,a)
                    t_sample = []
                    if t != 0:
                        t_sample.append(state_array[k])
                        t_sample.append(action.item())
                        trajectory_list_array[k].append(t_sample)
                
                self.EPSILON = self.EPSILON * self.EPSILON_DECAY
                
#               self.env.render()
                next_state_array, next_state_complete_array, reward_array, done_array, _ = self.env.step(action_array)
                
                for k in range(self.no_of_agents):
                    next_state_complete_array[k] = torch.from_numpy(next_state_complete_array[k]).view(1,self.n_states).to(device)
                    reward_array[k] = torch.tensor([reward_array[k]], dtype=torch.float32, device=device)
                    # Store the transition (s,a,s',r) in memory
                    self.replay_buffer.push(state_complete_array[k], action_array[k], next_state_complete_array[k], reward_array[k])

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                
                # Move to the next state
                state_complete_array = copy.deepcopy(next_state_complete_array)
                state_array = copy.deepcopy(next_state_array)
                for k in range(self.no_of_agents):
                    score_array[k] += reward_array[k].item()

                if ((sum(done_array) > 0) or (t+1 == t_max)):
                    # print("DQN-LunarLander: episode = {} | t = {} | score = {}".format(i_episode, t, round(score.item(),0)))
                    self.compute_approxQ()
                    for k in range(self.no_of_agents):
                        done_attack, accuracy, accuracy_softmax, accuracy_softmax_complete = utils_attack.Attack_Done_Identify(self.env, TARGET, self.approxQ_array[k])
                        accuracy_episode_array[k].append(accuracy)
                        accuracy_softmax_episode_array[k].append(accuracy_softmax)
                        accuracy_softmax_complete_episode_array[k].append(accuracy_softmax_complete)
                    break
                    
            # Trajectory to MEMORY with head_padding
            for k in range(self.no_of_agents):
                
                if len(trajectory_list_array[k]) < LEN_TRAJECTORY:
                    padding_state = 0 #torch.zeros(state.shape, device=device)
                    padding_action = 0 #torch.zeros(action.shape, device=device)
                    n_padding = LEN_TRAJECTORY - len(trajectory_list_array[k])
                    for i in range(n_padding):
                        self.MEM_array[k].push(padding_state, padding_action)
                    for i in range(len(trajectory_list_array[k])):
                        state = trajectory_list_array[k][i][0]
                        action = trajectory_list_array[k][i][1]
                        self.MEM_array[k].push(state, action)
                else:
                    for i in range(LEN_TRAJECTORY):
                        state = trajectory_list_array[k][i][0]
                        action = trajectory_list_array[k][i][1]
                        self.MEM_array[k].push(state, action)

            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return accuracy_episode_array, accuracy_softmax_episode_array, accuracy_softmax_complete_episode_array
                
                
if __name__ == "__main__":
    from envs.env3D_4x4_new_state_space_MA import GridWorld_3D_env
    
    no_of_agents = 2
    env = GridWorld_3D_env(no_of_agents)
    
    victim_args_dqn_pop = {
		"env": env, 
		"MEMORY_SIZE": 50, #5000, 
		"BATCH_SIZE": 512, #128, #8, #16, #32, #512, 
		"GAMMA": 0.99, 
		"EPS_START": 1.0, 
		"EPS_END": 0.01, 
		"EPS_DECAY": 0.996,
        "EPSILON": 1.0,
        "EPSILON_DECAY": 0.996,
		"TARGET_UPDATE": 10, #5, #10 
		"n_states": 32, #env.observation_space.shape[0], 
		"n_actions": 4, #env.action_space.n,
		"learning_rate": 0.001,
        "no_of_agents": no_of_agents
	}
    
    
    
    j = 1
    #score_array = []
    while(j > 0):
        victim_pop = VictimAgent_DQN_Pop(**victim_args_dqn_pop)
        episodes = 450
        t_max = 20
        victim_pop.train_model(episodes, t_max)
        score_array = victim_pop.eval_model()
        print("j: " + str(j) + " score: " + str(score_array) + " EPSILON: " + str(victim_pop.EPSILON))
        for k in range(no_of_agents):
            print("Agent: " + str(k))
            print(victim_pop.approxQ_data_array[k])
            print(victim_pop.approxQ_array[k])
            print(victim_pop.LA_array[k])
        #score_array.append([j, score.item()])
        j = j+1
        #victim.eval_model_memory()
        
        #victim.Train_Model(episodes)
        #victim.Show_PolicyQ()
        #victim.Eval_Model()
    
        #print(f"Size of Memory = {victim.MEM.__len__()}")
        #print(f"First 5 Trajectory is \n{victim.MEM.memory[0:5]}")
    
    #victim_args = {
        #"env": env, 
        #"MEMORY_SIZE": 50, #100, 
        #"discount_factor": 0.9, #1.0, 
        #"alpha": 0.1, 
        #"epsilon": 0.1,
    #}
    
    