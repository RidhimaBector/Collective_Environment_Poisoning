import os
from os.path import dirname, abspath
import sys
if "../" not in sys.path:
    sys.path.append("../") 
    
from victim.victim_Q import VictimAgent
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

class System():
    def __init__(self, victim_args, ae_args, no_of_agents):
        #self.victim = VictimAgent(**victim_args)
        self.no_of_agents = no_of_agents
        self.victim_array = []
        for _ in range(no_of_agents):
            self.victim_array.append(VictimAgent(**victim_args))
        self.ae = AutoEncoder(**ae_args)
        
    def train(self, victim_episodes_num = 50, ae_episodes_num = 1):
        self.ae.n_epochs = ae_episodes_num
        
        for i_episode in range(victim_episodes_num):
            
            for k in range(self.no_of_agents):
                
                self.victim_array[k].Train_Model(1)
                
                if self.victim_array[k].MEM.__len__() >= MEMORY_SIZE:
                    self.ae.Train(self.victim_array[k].MEM)
                
    def eval_victim(self):
        self.victim.Eval_Model()