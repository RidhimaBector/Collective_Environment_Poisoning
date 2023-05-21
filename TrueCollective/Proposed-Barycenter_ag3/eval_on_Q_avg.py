import numpy as np
import torch
import gym
import argparse
import os
import sys
import time
import copy
import ot

from collections import defaultdict
from itertools import count

# TensorBoard
import tensorboardX
import datetime

# Configuration
"""from yacs.config import CfgNode as CN
yaml_name='config/config_default.yaml'
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()"""

#SEQ_LEN = 6 #config.AE.SEQ_LEN
EMBEDDING_SIZE = 5 #config.AE.EMBEDDING_SIZE
MEMORY_SIZE = 50 #config.AE.MEMORY_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attack package
from utils import utils_buf, utils_op, utils_attack, utils_log
from attack.DDPG import DDPG
from envs.target_def import TARGET

no_of_agents = 3
# Environment object
from envs.env3D_4x4_MA_Setting3 import GridWorld_3D_env
env = GridWorld_3D_env(no_of_agents)
INIT_T = env.T.copy()

# Victim object
from victim.victim_DQN_new_state_space_pop_random import VictimAgent_DQN_Pop
#from victim.victim_Sarsa_eval import VictimAgent_Sarsa
#from victim.victim_MC_eval import VictimAgent_MC

# AutoEncoder
from ae.vae import VAE #from ae.ae import AutoEncoder

#Cost Matrix
grid = np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]])
cost_matrix = ot.dist(grid, grid, metric='cityblock') * 20
np.fill_diagonal(cost_matrix, 10)
cost_matrix = np.repeat(cost_matrix, env.nA, axis=1)
cost_matrix = np.repeat(cost_matrix, env.nA, axis=0)
np.fill_diagonal(cost_matrix, 0)

#victim_array = []
no_of_samples = 10
pm_embedding_dim = 5

''' ... PATH ... '''
policy_no = str(sys.argv[1])
PATH = "storage/R_0818/good_model_" + policy_no
PATH_vae = "storage/R_0818/" + "96984" + "_f-o_VAE" #"storage/R_0818/" + "340240" + "_f-o_AutoEncoder_SftMx" #"14800" + "_f-o_AutoEncoder"

# Set seeds
seed = 0
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

""" parameter of DDPG """

discount=0.95                # Discount factor
tau=0.005                    # Target network update rate

''' ..... Attack Network ..... '''
# Input / Output size
state_dim = EMBEDDING_SIZE + env.nS
action_dim = env.Attack_ActionSpace.shape[0]
max_action = float(env.Attack_ActionSpace.high[0])

attack_args = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": discount,
    "tau": tau,
}

kwargsNew = {
    "seed": 0,
    "nb_states": state_dim,
    "nb_actions": action_dim,
    "max_action": max_action,
    "hidden1": 400,
    "hidden2": 300,
    "init_w": 0.003,
    "prate": 0.0001,
    "rate": 0.001,
    "ou_theta": 0.15,
    "ou_mu": 0.0,
    "ou_sigma": 0.2,
    "bsize": 256,
    "tau": tau,
    "discount": discount,
    "epsilon_divisor": 1000, #Number of episodes over which to decrease actual epsilon
    "is_training": True
}

""" load Policy """ 
Policy = DDPG(**kwargsNew)
Policy.load(PATH)

''' ..... Victim ..... '''
victim_args = {
    "env": env, 
    "MEMORY_SIZE": MEMORY_SIZE,
    "discount_factor": 0.9, 
    "alpha": 0.1, 
    "epsilon": 0.1,
}

victim_args_dqn_pop = {
	"env": env, 
	"MEMORY_SIZE": MEMORY_SIZE, #5000, 
	"BATCH_SIZE": 512, #128, #8, #16, #32, #512, 
	"GAMMA": 0.99, 
	"EPS_START": 1.0, 
	"EPS_END": 0.01, 
	"EPS_DECAY": 0.996,
    "EPSILON": 1.0,
    "EPSILON_DECAY": 0.996,
	"TARGET_UPDATE": 10, #5, #10 
	"n_states": 33, #env.observation_space.shape[0], 
	"n_actions": 4, #env.action_space.n,
	"learning_rate": 0.001,
    "no_of_agents": no_of_agents
}

#victim = VictimAgent(**victim_args)
#for _ in range(no_of_agents):
    #victim_array.append(VictimAgent(**victim_args))
victim_pop = VictimAgent_DQN_Pop(**victim_args_dqn_pop)

""" parameter of VAE """
enc_in_size = 32 # 32 as 16 total states #SEQ_LEN*2 = 5*2
enc_out_size = 5 # EMBEDDING_SIZE = 5
dec_in_size = 6 #1+EMBEDDING_SIZE
dec_out_size = 5 #action_dim 0,1,2,3,4 - 4 for target states that victim has not visited yet

net_args = {
    "enc_in_size": enc_in_size, 
    "enc_out_size": enc_out_size, 
    "dec_in_size": dec_in_size, 
    "dec_out_size": dec_out_size, 
    "lr": 0.00001, #0.001,
}

vae = VAE(**net_args) #ae = AutoEncoder(**ae_args)
vae.load(PATH_vae) #ae.load(PATH_ae)

""" Func: evaluation """
Tmax = 15 #50 #100
victim_episodes = 40
Plot_Seed = "Same" #"3e-3"
Model = "Barycenter PopSize=3"
Train_Pop_Size = 3
n_victim_pop = 5 #10 #10
def eval_policy(policy):
    # log
    distance_K_atk_timestep = np.zeros((no_of_agents, Tmax)) #[] #complete data
    distance_grid_K_atk_timestep = np.zeros((no_of_agents, Tmax)) #[] #complete data
    distance_behavior_K_atk_timestep = np.zeros((no_of_agents, Tmax)) #[] #complete data
    distance_W_atk_timestep = np.zeros((no_of_agents, Tmax)) #[] #complete data
    distance_grid_W_atk_timestep = np.zeros((no_of_agents, Tmax)) #[] #complete data
    distance_behavior_W_atk_timestep = np.zeros((no_of_agents, Tmax)) #[] #complete data
    accuracy_victim_timestep = np.zeros((no_of_agents, Tmax*victim_episodes)) #[] #complete data
    accuracy_atk_timestep = np.zeros((no_of_agents, Tmax)) #[] #complete data
    accuracy_sftmx_victim_timestep = np.zeros((no_of_agents, Tmax*victim_episodes)) #[] #complete data
    accuracy_sftmx_atk_timestep = np.zeros((no_of_agents, Tmax)) #[] #complete data
    accuracy_sftmx_complete_victim_timestep = np.zeros((no_of_agents, Tmax*victim_episodes)) #[] #complete data
    accuracy_sftmx_complete_atk_timestep = np.zeros((no_of_agents, Tmax)) #[] #complete data
    effort_atk_timestep = [] #complete data
    time_atk_timestep = [] #complete data

    # reset victim's env and Q
    env.reset_altitude()
    #victim.reset()
    #for k in range(no_of_agents):
        #victim_array[k].reset()
    victim_pop.reset()

    # Initialize the attacker's state
    victim_info = np.zeros((1,EMBEDDING_SIZE))
    victim_tensor = torch.from_numpy(victim_info)
    victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

    env_info = env.altitude.copy()
    env_tensor = torch.from_numpy(env_info)
    env_tensor = env_tensor.view(1, env.nS)
    env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

    state = torch.cat((victim_tensor_4d, env_tensor_4d), 3)
    
    orgA = env.altitude.copy().reshape((16, 1)) #New
    curA =  orgA.copy()

    for t in range(Tmax):
        tic_timestep = time.time()

        # select action
        action = Policy.select_on_policy_action(np.array(state)) #Policy.select_action(np.array(state))

        ## perform attack_action on Env
        env.Attack_Env(action)
        
        # next_victim_info: compute victim's updated policy
        #accuracy_array, accuracy_sftmx_array, accuracy_sftmx_complete_array, victim_transitions = victim.train_for_eval(80)        
        #accuracy_array, accuracy_sftmx_array, accuracy_sftmx_complete_array, victim_transitions_array = [], [], [], []
        #for k in range(no_of_agents):
            #accuracy_array_T, accuracy_sftmx_array_T, accuracy_sftmx_complete_array_T, victim_transitions_T = victim_array[k].train_for_eval(80)
            #accuracy_array.append(accuracy_array_T)
            #accuracy_sftmx_array.append(accuracy_sftmx_array_T)
            #accuracy_sftmx_complete_array.append(accuracy_sftmx_complete_array_T)
            #victim_transitions_array.append(victim_transitions_T)
        accuracy_array, accuracy_sftmx_array, accuracy_sftmx_complete_array, victim_transitions_array = victim_pop.train_for_eval(num_episodes=victim_episodes, t_max=20)
        
        ### ... updated victim.Q
        #next_victim_info = vae.Embedding_Samples(victim_transitions, no_of_samples) #ae.Policy_Embedding(victim_transitions)
        #Barycenter
        measures_locations = [] #np.zeros((pm_no_of_agents*no_of_samples, pm_embedding_dim))
        measures_weights = []
        for k in range(no_of_agents):
            #measures_locations[k*no_of_samples:(k+1)*no_of_samples] = pm.Policy_Embedding(policy_model, enc_model, replay_buffer_victim_list, pm_batch_size, pm_embedding_dim, encoder_network_type, k, encoder_input, target_distribution, risk_strategy, no_of_samples).detach().numpy()
            with torch.no_grad():
                measures_locations.append(vae.Embedding_Samples(victim_transitions_array[k], no_of_samples))
            measures_weights.append(np.ones((no_of_samples,)) * (1/no_of_samples))
        barycenter_samples = 1  # number of Diracs/samples of the barycenter
        barycenter_initial_locations = np.random.normal(0., 1., (barycenter_samples, pm_embedding_dim))  # initial Dirac locations
        barycenter_weights = np.ones((barycenter_samples,)) / barycenter_samples  # weights of the barycenter (it will not be optimized, only the locations are optimized)
        barycenter = ot.lp.free_support_barycenter(measures_locations, measures_weights, barycenter_initial_locations, barycenter_weights)
        next_victim_info = barycenter #torch.FloatTensor(barycenter, device="cpu") #tensor

        next_victim_tensor = torch.from_numpy(next_victim_info[-1]).unsqueeze(0)
        next_victim_tensor_4d = next_victim_tensor.unsqueeze(0).unsqueeze(0)
        ### ... updated env altitude
        next_env_info = env.altitude.copy()
        next_env_tensor = torch.from_numpy(next_env_info)
        next_env_tensor = next_env_tensor.view(1, env.nS)
        next_env_tensor_4d = next_env_tensor.unsqueeze(0).unsqueeze(0)
        ### ... next_state
        next_state = torch.cat((next_victim_tensor_4d, next_env_tensor_4d), 3)
        
        # Step: cost
        distance_K, distance_grid_K, distance_behavior_K, distance_W, distance_grid_W, distance_behavior_W = [], [], [], [], [], []
        for k in range(no_of_agents):
            distance_K.append(utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim_pop.approxQ_array[k], TARGET, cost_matrix, distance_type=0)) #system.victim.Q
            distance_grid_K.append(utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim_pop.approxQ_array[k], TARGET, cost_matrix, distance_type=1))
            distance_behavior_K.append(utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim_pop.approxQ_array[k], TARGET, cost_matrix, distance_type=2))
            distance_W.append(utils_attack.Attack_Cost_Compute_W(env, INIT_T, victim_pop.approxQ_array[k], TARGET, cost_matrix, distance_type=0)) #system.victim.Q
            distance_grid_W.append(utils_attack.Attack_Cost_Compute_W(env, INIT_T, victim_pop.approxQ_array[k], TARGET, cost_matrix, distance_type=1))
            distance_behavior_W.append(utils_attack.Attack_Cost_Compute_W(env, INIT_T, victim_pop.approxQ_array[k], TARGET, cost_matrix, distance_type=2))
            
        #accuracy_array, accuracy_sftmx_array, accuracy_sftmx_complete_array = utils_attack.Attack_Done_Identify(env, TARGET, victim.Q) #system.victim.Q)
        effort, curA = utils_attack.Attack_Effort(curA, env)
        #effort = - effort
        toc_timestep = time.time()
        time_timestep = toc_timestep - tic_timestep #tic_timestep - toc_timestep #- (toc_timestep - tic_timestep)
        
        # Step: log
        for k in range(no_of_agents):
            distance_K_atk_timestep[k, t] = distance_K[k]
            distance_grid_K_atk_timestep[k, t] = distance_grid_K[k]
            distance_behavior_K_atk_timestep[k, t] = distance_behavior_K[k]
            distance_W_atk_timestep[k, t] = distance_W[k]
            distance_grid_W_atk_timestep[k, t] = distance_grid_W[k]
            distance_behavior_W_atk_timestep[k, t] = distance_behavior_W[k]
            accuracy_victim_timestep[k, t*victim_episodes : (t+1)*victim_episodes] = np.array(accuracy_array[k])
            accuracy_atk_timestep[k, t] = accuracy_array[k][-1]
            accuracy_sftmx_victim_timestep[k, t*victim_episodes : (t+1)*victim_episodes] = np.array(accuracy_sftmx_array[k])
            accuracy_sftmx_atk_timestep[k, t] = accuracy_sftmx_array[k][-1]
            accuracy_sftmx_complete_victim_timestep[k, t*victim_episodes : (t+1)*victim_episodes] = np.array(accuracy_sftmx_complete_array[k])
            accuracy_sftmx_complete_atk_timestep[k, t] = accuracy_sftmx_complete_array[k][-1]
        
        effort_atk_timestep.append(effort)
        time_atk_timestep.append(time_timestep)

        # update state 
        state = copy.deepcopy(next_state)

        #if done:
            #break

    #print("---------------------------------------")
    #cum_reward = cumulative_reward #/(t+1)
    #print(f"Evaluation over {t} timesteps: {cumulative_reward:.3f}")
    #utils_op.Show_PolicyQ(victim.Q, env)
    #print("---------------------------------------")


    return distance_K_atk_timestep.flatten(), distance_grid_K_atk_timestep.flatten(), distance_behavior_K_atk_timestep.flatten(), distance_W_atk_timestep.flatten(), distance_grid_W_atk_timestep.flatten(), distance_behavior_W_atk_timestep.flatten(), accuracy_victim_timestep, accuracy_atk_timestep.flatten(), accuracy_sftmx_victim_timestep, accuracy_sftmx_atk_timestep.flatten(), accuracy_sftmx_complete_victim_timestep, accuracy_sftmx_complete_atk_timestep.flatten(), effort_atk_timestep, time_atk_timestep



""" Evaluate """
distance_K_atk_timestep_cluster = [] #complete data
distance_grid_K_atk_timestep_cluster = [] #complete data
distance_behavior_K_atk_timestep_cluster = [] #complete data
distance_W_atk_timestep_cluster = [] #complete data
distance_grid_W_atk_timestep_cluster = [] #complete data
distance_behavior_W_atk_timestep_cluster = [] #complete data
accuracy_victim_timestep_cluster = [] #complete data
accuracy_atk_timestep_cluster = [] #complete data
accuracy_sftmx_victim_timestep_cluster = [] #complete data
accuracy_sftmx_atk_timestep_cluster = [] #complete data
accuracy_sftmx_complete_victim_timestep_cluster = [] #complete data
accuracy_sftmx_complete_atk_timestep_cluster = [] #complete data
effort_atk_timestep_cluster = [] #complete data
time_atk_timestep_cluster = [] #complete data


for i_victim_pop in range(n_victim_pop):
    distance_K_atk_timestep, distance_grid_K_atk_timestep, distance_behavior_K_atk_timestep, distance_W_atk_timestep, distance_grid_W_atk_timestep, distance_behavior_W_atk_timestep, accuracy_victim_timestep, accuracy_atk_timestep, accuracy_sftmx_victim_timestep, accuracy_sftmx_atk_timestep, accuracy_sftmx_complete_victim_timestep, accuracy_sftmx_complete_atk_timestep, effort_atk_timestep, time_atk_timestep = eval_policy(Policy)

    distance_K_atk_timestep_cluster.append(distance_K_atk_timestep)
    distance_grid_K_atk_timestep_cluster.append(distance_grid_K_atk_timestep)
    distance_behavior_K_atk_timestep_cluster.append(distance_behavior_K_atk_timestep)
    distance_W_atk_timestep_cluster.append(distance_W_atk_timestep)
    distance_grid_W_atk_timestep_cluster.append(distance_grid_W_atk_timestep)
    distance_behavior_W_atk_timestep_cluster.append(distance_behavior_W_atk_timestep)
    accuracy_victim_timestep_cluster.append(accuracy_victim_timestep)
    accuracy_atk_timestep_cluster.append(accuracy_atk_timestep)
    accuracy_sftmx_victim_timestep_cluster.append(accuracy_sftmx_victim_timestep)
    accuracy_sftmx_atk_timestep_cluster.append(accuracy_sftmx_atk_timestep)
    accuracy_sftmx_complete_victim_timestep_cluster.append(accuracy_sftmx_complete_victim_timestep)
    accuracy_sftmx_complete_atk_timestep_cluster.append(accuracy_sftmx_complete_atk_timestep)
    effort_atk_timestep_cluster.append(effort_atk_timestep)
    time_atk_timestep_cluster.append(time_atk_timestep)

    
""" unify length """
max_size = Tmax * victim_episodes #4800 #2500 #5000 #Tmax * 80

"""for i in range(n_victim):
    last_value = Rate_List[i][-1]
    print(last_value)
    if len(Rate_List[i]) < max_size:
        delta_size = max_size - len(Rate_List[i])
        for j in range(delta_size):
            Rate_List[i].append(last_value)"""
            
            
""" average """
avg_step = 5 #25
accuracy_victim_timestep_avg_cluster = []
accuracy_sftmx_victim_timestep_avg_cluster = []
accuracy_sftmx_complete_victim_timestep_avg_cluster = []

N = max_size
N_avg = N//avg_step

for i_victim_pop in range(n_victim_pop):
    accuracy_victim_timestep_avg_temp = []
    accuracy_sftmx_victim_timestep_avg_temp = []
    accuracy_sftmx_complete_victim_timestep_avg_temp = []
    
    for k in range(no_of_agents):
        accuracy_victim_timestep_avg_temp.append(0)
        accuracy_sftmx_victim_timestep_avg_temp.append(0)
        accuracy_sftmx_complete_victim_timestep_avg_temp.append(0)
    
        for i in range(0, N_avg):
            start = i*avg_step
            end = (i+1)*avg_step
    
            tmp_acc = sum(accuracy_victim_timestep_cluster[i_victim_pop][k][start: end])
            tmp_acc_sftmx = sum(accuracy_sftmx_victim_timestep_cluster[i_victim_pop][k][start: end])
            tmp_acc_sftmx_c = sum(accuracy_sftmx_complete_victim_timestep_cluster[i_victim_pop][k][start: end])
            
            accuracy_victim_timestep_avg_temp.append(tmp_acc/avg_step)
            accuracy_sftmx_victim_timestep_avg_temp.append(tmp_acc_sftmx/avg_step)
            accuracy_sftmx_complete_victim_timestep_avg_temp.append(tmp_acc_sftmx_c/avg_step)
        
    accuracy_victim_timestep_avg_cluster.append(accuracy_victim_timestep_avg_temp)
    accuracy_sftmx_victim_timestep_avg_cluster.append(accuracy_sftmx_victim_timestep_avg_temp)
    accuracy_sftmx_complete_victim_timestep_avg_cluster.append(accuracy_sftmx_complete_victim_timestep_avg_temp)
    #print(len(per_avg_rate))
    #print(np.round(per_avg_rate,2))
            
            
""" dataframe """
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
sns.set(font_scale = 1.8)
sns.set_style("whitegrid")

time_compression = np.arange( len(accuracy_victim_timestep_avg_cluster[0])/no_of_agents ) #list(range(0, len(avg_rate[0])))
vic_time =  np.tile((time_compression * avg_step) / victim_episodes, no_of_agents) #list(np.array(time_compression) * avg_step)
atk_time = np.tile(np.arange(1,Tmax+1), no_of_agents)
ag_number = np.arange(no_of_agents)
vic_ag_number = np.repeat(ag_number, N_avg+1)
atk_ag_number = np.repeat(ag_number, Tmax)
vic_Pop_name = ["VP1", "VP2", "VP3", "VP4", "VP5", "VP6", "VP7", "VP8", "VP9", "VP10", "VP11", "VP12", "VP13", "VP14", "VP15", "VP16", "VP17", "VP18", "VP19", "VP20"]

df_acc_V = pd.DataFrame({"Victim Timestep":vic_time, "Model":Model, "Accuracy V":accuracy_victim_timestep_avg_cluster[0], "Sftmx Accuracy V":accuracy_sftmx_victim_timestep_avg_cluster[0], "C Sftmc Accuracy V":accuracy_sftmx_complete_victim_timestep_avg_cluster[0], "Victim":vic_ag_number, "Victim Pop":vic_Pop_name[0], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
df_acc_A = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Accuracy A":accuracy_atk_timestep_cluster[0], "Sftmx Accuracy A":accuracy_sftmx_atk_timestep_cluster[0], "C Sftmx Accuracy A":accuracy_sftmx_complete_atk_timestep_cluster[0], "Victim":atk_ag_number, "Victim Pop":vic_Pop_name[0], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
df_dis_K = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Distance":distance_K_atk_timestep_cluster[0], "Grid Distance":distance_grid_K_atk_timestep_cluster[0], "Behavior Distance":distance_behavior_K_atk_timestep_cluster[0], "Victim":atk_ag_number, "Victim Pop":vic_Pop_name[0], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
df_dis_W = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Distance":distance_W_atk_timestep_cluster[0], "Grid Distance":distance_grid_W_atk_timestep_cluster[0], "Behavior Distance":distance_behavior_W_atk_timestep_cluster[0], "Victim":atk_ag_number, "Victim Pop":vic_Pop_name[0], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
df_eff = pd.DataFrame({"Attacker Timestep":np.arange(1,Tmax+1), "Model":Model, "Effort":effort_atk_timestep_cluster[0], "Attack Timestep Time":time_atk_timestep_cluster[0], "Victim Pop":vic_Pop_name[0], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
for i in range(1, n_victim_pop):
    tmp_df_acc_V = pd.DataFrame({"Victim Timestep":vic_time, "Model":Model, "Accuracy V":accuracy_victim_timestep_avg_cluster[i], "Sftmx Accuracy V":accuracy_sftmx_victim_timestep_avg_cluster[i], "C Sftmc Accuracy V":accuracy_sftmx_complete_victim_timestep_avg_cluster[i], "Victim":vic_ag_number, "Victim Pop":vic_Pop_name[i], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
    tmp_df_acc_A = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Accuracy A":accuracy_atk_timestep_cluster[i], "Sftmx Accuracy A":accuracy_sftmx_atk_timestep_cluster[i], "C Sftmx Accuracy A":accuracy_sftmx_complete_atk_timestep_cluster[i], "Victim":atk_ag_number, "Victim Pop":vic_Pop_name[i], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
    tmp_df_dis_K = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Distance":distance_K_atk_timestep_cluster[i], "Grid Distance":distance_grid_K_atk_timestep_cluster[i], "Behavior Distance":distance_behavior_K_atk_timestep_cluster[i], "Victim":atk_ag_number, "Victim Pop":vic_Pop_name[i], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
    tmp_df_dis_W = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Distance":distance_W_atk_timestep_cluster[i], "Grid Distance":distance_grid_W_atk_timestep_cluster[i], "Behavior Distance":distance_behavior_W_atk_timestep_cluster[i], "Victim":atk_ag_number, "Victim Pop":vic_Pop_name[i], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
    tmp_df_eff = pd.DataFrame({"Attacker Timestep":np.arange(1,Tmax+1), "Model":Model, "Effort":effort_atk_timestep_cluster[i], "Attack Timestep Time":time_atk_timestep_cluster[i], "Victim Pop":vic_Pop_name[i], "Train Pop Size":Train_Pop_Size ,"Test Pop Size":no_of_agents, "Seed":Plot_Seed })
    df_acc_V = df_acc_V.append(tmp_df_acc_V, ignore_index=True)
    df_acc_A = df_acc_A.append(tmp_df_acc_A, ignore_index=True)
    df_dis_K = df_dis_K.append(tmp_df_dis_K, ignore_index=True)
    df_dis_W = df_dis_W.append(tmp_df_dis_W, ignore_index=True)
    df_eff = df_eff.append(tmp_df_eff, ignore_index=True)
df_acc_V.to_csv(policy_no + "_accuracy_V_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".csv", index=False)
df_acc_A.to_csv(policy_no + "_accuracy_A_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".csv", index=False)
df_dis_K.to_csv(policy_no + "_distance_K_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".csv", index=False)
df_dis_W.to_csv(policy_no + "_distance_W_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".csv", index=False)
df_eff.to_csv(policy_no + "_effort_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".csv", index=False)

""" Figure & Data """
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
sns_plot_acc_V = sns.lineplot(x = "Victim Timestep", y = "Accuracy V", hue="Model", data=df_acc_V, ax=axs[0,0])
sns_plot_acc_A = sns.lineplot(x = "Attacker Timestep", y = "Accuracy A", hue="Model", data=df_acc_A, ax=axs[0,1])
sns_plot_acc_sft_V = sns.lineplot(x = "Victim Timestep", y = "Sftmx Accuracy V", hue="Model", data=df_acc_V, ax=axs[1,0])
sns_plot_acc_sft_A = sns.lineplot(x = "Attacker Timestep", y = "Sftmx Accuracy A", hue="Model", data=df_acc_A, ax=axs[1,1])
sns_plot_acc_sft_C_V = sns.lineplot(x = "Victim Timestep", y = "C Sftmc Accuracy V", hue="Model", data=df_acc_V, ax=axs[2,0])
sns_plot_acc_sft_C_A = sns.lineplot(x = "Attacker Timestep", y = "C Sftmx Accuracy A", hue="Model", data=df_acc_A, ax=axs[2,1])
plt.savefig(policy_no + "_accuracy_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".png")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
sns_plot_dis_grid_K = sns.lineplot(x = "Attacker Timestep", y = "Grid Distance", hue="Model", data=df_dis_K, ax=axs[0,0])
sns_plot_dis_beh_K = sns.lineplot(x = "Attacker Timestep", y = "Behavior Distance", hue="Model", data=df_dis_K, ax=axs[0,1])
sns_plot_dis_K = sns.lineplot(x = "Attacker Timestep", y = "Distance", hue="Model", data=df_dis_K, ax=axs[1,0])
sns_plot_time = sns.lineplot(x = "Attacker Timestep", y = "Attack Timestep Time", hue="Model", data=df_eff, ax=axs[1,1])
plt.savefig(policy_no + "_distance_K_time_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".png")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
sns_plot_dis_grid_W = sns.lineplot(x = "Attacker Timestep", y = "Grid Distance", hue="Model", data=df_dis_W, ax=axs[0,0])
sns_plot_dis_beh_W = sns.lineplot(x = "Attacker Timestep", y = "Behavior Distance", hue="Model", data=df_dis_W, ax=axs[0,1])
sns_plot_dis_W = sns.lineplot(x = "Attacker Timestep", y = "Distance", hue="Model", data=df_dis_W, ax=axs[1,0])
sns_plot_eff = sns.lineplot(x = "Attacker Timestep", y = "Effort", hue="Model", data=df_eff, ax=axs[1,1])
plt.savefig(policy_no + "_distance_W_effort_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".png")

#Separate line for each population
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
sns_plot_acc_V = sns.lineplot(x = "Victim Timestep", y = "Accuracy V", hue="Victim Pop", data=df_acc_V, ax=axs[0,0])
sns_plot_acc_A = sns.lineplot(x = "Attacker Timestep", y = "Accuracy A", hue="Victim Pop", data=df_acc_A, ax=axs[0,1])
sns_plot_acc_sft_V = sns.lineplot(x = "Victim Timestep", y = "Sftmx Accuracy V", hue="Victim Pop", data=df_acc_V, ax=axs[1,0])
sns_plot_acc_sft_A = sns.lineplot(x = "Attacker Timestep", y = "Sftmx Accuracy A", hue="Victim Pop", data=df_acc_A, ax=axs[1,1])
sns_plot_acc_sft_C_V = sns.lineplot(x = "Victim Timestep", y = "C Sftmc Accuracy V", hue="Victim Pop", data=df_acc_V, ax=axs[2,0])
sns_plot_acc_sft_C_A = sns.lineplot(x = "Attacker Timestep", y = "C Sftmx Accuracy A", hue="Victim Pop", data=df_acc_A, ax=axs[2,1])
plt.savefig(policy_no + "_accuracy_P_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".png")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
sns_plot_dis_grid_K = sns.lineplot(x = "Attacker Timestep", y = "Grid Distance", hue="Victim Pop", data=df_dis_K, ax=axs[0,0])
sns_plot_dis_beh_K = sns.lineplot(x = "Attacker Timestep", y = "Behavior Distance", hue="Victim Pop", data=df_dis_K, ax=axs[0,1])
sns_plot_dis_K = sns.lineplot(x = "Attacker Timestep", y = "Distance", hue="Victim Pop", data=df_dis_K, ax=axs[1,0])
sns_plot_time = sns.lineplot(x = "Attacker Timestep", y = "Attack Timestep Time", hue="Victim Pop", data=df_eff, ax=axs[1,1])
plt.savefig(policy_no + "_distance_K_time_P_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".png")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
sns_plot_dis_grid_W = sns.lineplot(x = "Attacker Timestep", y = "Grid Distance", hue="Victim Pop", data=df_dis_W, ax=axs[0,0])
sns_plot_dis_beh_W = sns.lineplot(x = "Attacker Timestep", y = "Behavior Distance", hue="Victim Pop", data=df_dis_W, ax=axs[0,1])
sns_plot_dis_W = sns.lineplot(x = "Attacker Timestep", y = "Distance", hue="Victim Pop", data=df_dis_W, ax=axs[1,0])
sns_plot_eff = sns.lineplot(x = "Attacker Timestep", y = "Effort", hue="Victim Pop", data=df_eff, ax=axs[1,1])
plt.savefig(policy_no + "_distance_W_effort_P_" + Model + f"_Train{Train_Pop_Size}_Test{no_of_agents}_Seed" + Plot_Seed + ".png")


# plt.show() # to show graph
#fig = sns_plot.get_figure()
#fig.savefig(TITLE_figure, bbox_inches='tight')