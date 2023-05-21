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
#from victim.system import System
from victim.victim_DQN_new_state_space_pop_random import VictimAgent_DQN_Pop
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_dir", default="R_0818")               # TensorBoard folder
    
    parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    #parser.add_argument("--atk_training_start_episodes", default=100, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eps_greedy_start_episodes", default=30, type=int)  # Time steps initial random policy is used
    parser.add_argument("--max_timesteps", default=15, type=int)   # Max time steps to run environment
    parser.add_argument("--max_episodes_num", default=30000, type=int)   # Max episodes to run environment
    parser.add_argument("--eval_freq_episode", default=20, type=int)        # How often (time steps) we evaluate
    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.95, type=float)     # Discount factor of attack network
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
    
    parser.add_argument("--victim_n_episodes", default=80, type=int)  # number of episodes for victim's updated in poisoned Env
    parser.add_argument("--ae_n_epochs", default=10, type=int)         # number of training epoch
    
    args = parser.parse_args()

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' ..... Tensorboard Settings ..... '''

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"results_{date}"
    
    model_name = args.model_dir or default_model_name
    model_dir = utils_log.get_model_dir(model_name)
    
    ''' ..... Attack Network ..... '''
    # Input / Output size
    state_dim = EMBEDDING_SIZE + env.nS
    action_dim = env.Attack_ActionSpace.shape[0]
    max_action = float(env.Attack_ActionSpace.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    
    kwargsNew = {
        "seed": args.seed,
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
        "tau": args.tau,
        "discount": args.discount,
        "epsilon_divisor": 1000, #Number of episodes over which to decrease actual epsilon
        "is_training": True
    }

    # Initialize Policy
    Policy = DDPG(**kwargsNew)
    Buffer = utils_buf.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    
    ''' ..... Victim ..... '''
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": MEMORY_SIZE,
        "discount_factor": 0.9, #1.0, 
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
    vae.load("storage/R_0818/" + "96984" + "_f-o_VAE") #ae.load("storage/R_0818/" + "340240" + "_f-o_AutoEncoder_SftMx") #"14800" + "_f-o_AutoEncoder") #44780 - p-o M/H 
    
    #system = System(victim_args, ae_args)

    atk_n_epoch = 1 #15
    atk_n_batch = 15
    ddpg_loss = []
    distance_buffer = [] #complete data
    distance_grid_buffer = [] #complete data
    distance_behavior_buffer = [] #complete data
    accuracy_buffer = [] #complete data
    accuracy_sftmx_buffer = [] #complete data
    accuracy_sftmx_complete_buffer = [] #complete data
    effort_buffer = [] #complete data
    time_buffer = [] #complete data
    model_data = [] #per episode statistic wrt all metrics
    model_good_data = [] #all model statistic for best models
    model_bad_data = [] #all model statistic for worst models
    
    best_model_stats = np.zeros((24,1)) #best statistics so far (across diff models)
    worst_model_stats = np.zeros((24,1)) #worst statistics so far (across diff models)

    ''' ..... Training ..... '''

    for i_episode in range(args.max_episodes_num):
        print(f"\n--------- Episode: {i_episode} ----------") #txt_logger.info(f"\n--------- Episode: {i_episode} ----------")
        cumulative_distance = np.zeros((1+no_of_agents)) #[0]*(1+no_of_agents) #cumulative over episode
        cumulative_distance_grid = np.zeros((1+no_of_agents))
        cumulative_distance_behavior = np.zeros((1+no_of_agents))
        cumulative_accuracy = np.zeros((1+no_of_agents))
        cumulative_accuracy_sftmx = np.zeros((1+no_of_agents))
        cumulative_accuracy_sftmx_complete = np.zeros((1+no_of_agents))
        cumulative_effort = 0
        cumulative_time = 0
        
        # reset victim's env and Q
        env.reset_altitude()
        #victim.reset() #victim_Q = np.zeros((16,4)) #system.victim.reset()
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

        x = torch.cat((victim_tensor_4d, env_tensor_4d), 3)
        curA = env.altitude.copy().reshape((16, 1))
        
        for t in range(args.max_timesteps):
            tic_timestep = time.time()

            # Select attack_action
            if i_episode < args.eps_greedy_start_episodes:
                u = env.Attack_ActionSpace.sample()
            else:
                u = Policy.select_ddpg_action(np.array(x))

            # Step: implement attack_action
            env.Attack_Env(u)

            # Step: victim updates = get next_x
            #victim_transitions = victim.Train_Model(80) #system.train(args.victim_n_episodes, args.ae_n_epochs)
            #victim_transitions_array = []
            #for k in range(no_of_agents):
                #victim_transitions_array.append(victim_array[k].Train_Model(80))
            victim_transitions_array = victim_pop.train_model(num_episodes=40, t_max=20)
            
            #next_victim_info = vae.Embedding_Samples(victim_transitions, no_of_samples) #ae.Policy_Embedding(victim_transitions) #next_victim_info = system.ae.Embedding(system.victim.MEM)
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
            next_env_info = env.altitude.copy() #system.victim.env.altitude.copy()
            next_env_tensor = torch.from_numpy(next_env_info)
            next_env_tensor = next_env_tensor.view(1, env.nS)
            next_env_tensor_4d = next_env_tensor.unsqueeze(0).unsqueeze(0)
            ### ... next_state
            next_x = torch.cat((next_victim_tensor_4d, next_env_tensor_4d), 3)
            
            # Step: cost
            distance, distance_grid, distance_behavior, done, accuracy, accuracy_sftmx, accuracy_sftmx_complete  = (np.zeros((1+no_of_agents)) for _ in range(7))
            for k in range(no_of_agents):           
                distance[k+1] = - utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim_pop.approxQ_array[k], TARGET, cost_matrix, distance_type=0) #system.victim.Q
                distance_grid[k+1] = - utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim_pop.approxQ_array[k], TARGET, cost_matrix, distance_type=1)
                distance_behavior[k+1] = - utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim_pop.approxQ_array[k], TARGET, cost_matrix, distance_type=2)
                done_T, accuracy_T, accuracy_sftmx_T, accuracy_sftmx_complete_T = utils_attack.Attack_Done_Identify(env, TARGET, victim_pop.approxQ_array[k]) #system.victim.Q)
                done[k+1] = done_T
                accuracy[k+1] = accuracy_T
                accuracy_sftmx[k+1] = accuracy_sftmx_T
                accuracy_sftmx_complete[k+1] = accuracy_sftmx_complete_T
            effort, curA = utils_attack.Attack_Effort(curA, env)
            effort = - effort
            toc_timestep = time.time()
            time_timestep = tic_timestep - toc_timestep #- (toc_timestep - tic_timestep)

            avg_distance = np.sum(distance)/no_of_agents
            avg_distance_grid = np.sum(distance_grid)/no_of_agents
            avg_distance_behavior = np.sum(distance_behavior)/no_of_agents
            done = np.prod(done[1:])
            avg_accuracy = np.sum(accuracy)/no_of_agents
            avg_accuracy_sftmx = np.sum(accuracy_sftmx)/no_of_agents
            avg_accuracy_sftmx_complete = np.sum(accuracy_sftmx_complete)/no_of_agents
            
            ### log
            #txt_logger.info(f"episode = {i_episode+1} | t = {t+1} | reward: {reward:.4f} | done: {done} | success: {rate:.3f}")
            no_episodes = i_episode+1
            no_timesteps = t+1
            distance[0] = avg_distance
            distance_grid[0] = avg_distance_grid
            distance_behavior[0] = avg_distance_behavior
            accuracy[0] = avg_accuracy
            accuracy_sftmx[0] = avg_accuracy_sftmx
            accuracy_sftmx_complete[0] = avg_accuracy_sftmx_complete     
            
            distance_buffer.append([no_episodes, no_timesteps])
            distance_buffer[-1].extend(distance.tolist())
            distance_grid_buffer.append([no_episodes, no_timesteps])
            distance_grid_buffer[-1].extend(distance_grid.tolist())
            distance_behavior_buffer.append([no_episodes, no_timesteps])
            distance_behavior_buffer[-1].extend(distance_behavior.tolist())
            accuracy_buffer.append([no_episodes, no_timesteps])
            accuracy_buffer[-1].extend(accuracy.tolist())
            accuracy_sftmx_buffer.append([no_episodes, no_timesteps])
            accuracy_sftmx_buffer[-1].extend(accuracy_sftmx.tolist())
            accuracy_sftmx_complete_buffer.append([no_episodes, no_timesteps])
            accuracy_sftmx_complete_buffer[-1].extend(accuracy_sftmx_complete.tolist())
            effort_buffer.append([no_episodes, no_timesteps, effort])
            time_buffer.append([no_episodes, no_timesteps, time_timestep])
            
            reward = avg_distance
            cumulative_distance += distance
            cumulative_distance_grid += distance_grid
            cumulative_distance_behavior += distance_behavior
            cumulative_accuracy += accuracy
            cumulative_accuracy_sftmx += accuracy_sftmx
            cumulative_accuracy_sftmx_complete += accuracy_sftmx_complete
            cumulative_effort += effort
            cumulative_time += time_timestep

            # Replay buffer
            Buffer.add(x.view(state_dim), u, next_x.view(state_dim), reward, done)

            # Update state
            x = copy.deepcopy(next_x)
            if (done or (no_timesteps == args.max_timesteps)):
                
                cur_model_stats = np.array([avg_distance, cumulative_distance[0]/no_timesteps, cumulative_distance[0], \
                                            avg_distance_grid, cumulative_distance_grid[0]/no_timesteps, cumulative_distance_grid[0], \
                                            avg_distance_behavior, cumulative_distance_behavior[0]/no_timesteps, cumulative_distance_behavior[0], \
                                            avg_accuracy, cumulative_accuracy[0]/no_timesteps, cumulative_accuracy[0], \
                                            avg_accuracy_sftmx, cumulative_accuracy_sftmx[0]/no_timesteps, cumulative_accuracy_sftmx[0], \
                                            avg_accuracy_sftmx_complete, cumulative_accuracy_sftmx_complete[0]/no_timesteps, cumulative_accuracy_sftmx_complete[0], \
                                            effort, cumulative_effort/no_timesteps, cumulative_effort, \
                                            time_timestep, cumulative_time/no_timesteps, cumulative_time])
                 
                model_data.append([no_episodes, no_timesteps])
                model_data[-1].extend(cur_model_stats.tolist())
                model_data[-1].extend(cumulative_distance[1:].tolist())
                model_data[-1].extend(cumulative_distance_grid[1:].tolist())
                model_data[-1].extend(cumulative_distance_behavior[1:].tolist())
                model_data[-1].extend(cumulative_accuracy[1:].tolist())
                model_data[-1].extend(cumulative_accuracy_sftmx[1:].tolist())
                model_data[-1].extend(cumulative_accuracy_sftmx_complete[1:].tolist())
                
                if(i_episode == 0):
                    best_model_stats = copy.deepcopy(cur_model_stats)
                    worst_model_stats = copy.deepcopy(cur_model_stats)
                    break
                
                cur_model_is_good = cur_model_stats >= best_model_stats
                cur_model_is_bad = cur_model_stats <= worst_model_stats
                
                if(np.sum(cur_model_is_good) > 0):
                    model_good_data.append([no_episodes, no_timesteps])
                    model_good_data[-1].extend(cur_model_stats.tolist())
                    model_good_data[-1].extend(best_model_stats.tolist())
                    model_good_data[-1].extend(worst_model_stats.tolist())
                    Policy.save(f"./{model_dir}/good_model_{no_episodes}")
                    
                    best_model_stats[cur_model_is_good] = cur_model_stats[cur_model_is_good]
                    
                elif(np.sum(cur_model_is_bad) > 0):
                    model_bad_data.append([no_episodes, no_timesteps])
                    model_bad_data[-1].extend(cur_model_stats.tolist())
                    model_bad_data[-1].extend(best_model_stats.tolist())
                    model_bad_data[-1].extend(worst_model_stats.tolist())
                    Policy.save(f"./{model_dir}/bad_model_{no_episodes}")
                    
                    worst_model_stats[cur_model_is_bad] = cur_model_stats[cur_model_is_bad]

                break

        # Attack_Policy Update
        if i_episode >= args.eps_greedy_start_episodes:
            ddpg_loss = Policy.train(Buffer, atk_n_epoch, atk_n_batch, args.batch_size, ddpg_loss, i_episode)      
        ''' save Attack_Policy '''
        if no_episodes % args.eval_freq_episode == 0:
            Buffer.saveBuffer(f"./{model_dir}/")
            Policy.save(f"./{model_dir}/{no_episodes}")
            np.savetxt("ddpg_loss.csv", np.array(ddpg_loss), delimiter=",")
            np.savetxt("distance_buffer.csv", np.array(distance_buffer), delimiter=",")
            np.savetxt("distance_grid_buffer.csv", np.array(distance_grid_buffer), delimiter=",")
            np.savetxt("distance_behavior_buffer.csv", np.array(distance_behavior_buffer), delimiter=",")
            np.savetxt("accuracy_buffer.csv", np.array(accuracy_buffer), delimiter=",")
            np.savetxt("accuracy_sftmx_buffer.csv", np.array(accuracy_sftmx_buffer), delimiter=",")
            np.savetxt("accuracy_sftmx_complete_buffer.csv", np.array(accuracy_sftmx_complete_buffer), delimiter=",")
            np.savetxt("effort_buffer.csv", np.array(effort_buffer), delimiter=",")
            np.savetxt("time_buffer.csv", np.array(time_buffer), delimiter=",")
            np.savetxt("model_data.csv", np.array(model_data), delimiter=",")
            np.savetxt("model_good_data.csv", np.array(model_good_data), delimiter=",")
            np.savetxt("model_bad_data.csv", np.array(model_bad_data), delimiter=",")