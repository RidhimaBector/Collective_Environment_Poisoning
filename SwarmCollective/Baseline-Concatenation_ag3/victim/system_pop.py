import os
from os.path import dirname, abspath
import sys
if "../" not in sys.path:
    sys.path.append("../") 
    
from victim.victim_DQN_new_state_space_pop import VictimAgent_DQN_Pop
from ae.ae_policy import AutoEncoder

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE



''' import target_Memory '''
from envs.target_def import TARGET

class System_Pop():
    def __init__(self, victim_args_dqn_pop, ae_args, no_of_agents):
        self.victim_pop = VictimAgent_DQN_Pop(**victim_args_dqn_pop) #self.victim = VictimAgent(**victim_args)
        self.ae = AutoEncoder(**ae_args)
        self.no_of_agents = no_of_agents
        
    def train(self, victim_episodes_num = 50, ae_episodes_num = 1):
        self.ae.n_epochs = ae_episodes_num
        
        for i_episode in range(victim_episodes_num):
            
            self.victim_pop.train_model(num_episodes=1, t_max=20) #self.victim.Train_Model(1)
            
            for k in range(self.no_of_agents):
                if self.victim_pop.MEM_array[k].__len__() >= MEMORY_SIZE:
                    self.ae.Train(self.victim_pop.MEM_array[k])
                
    def eval_victim(self):
        self.victim_pop.eval_model()