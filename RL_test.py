from env import Ur5
from DDPG_pytorch import DDPG
import numpy as np 
import matplotlib.pyplot as plt
from math import pi
import torch

TRAIN_CONFIG = {'state_dim':10,'action_dim':5,'action_bound':pi/36,'test_epoch':100}
PATH_TO_MODEL = '/home/waiyang/pana_RL_yueci/model_para/'
MODEL_DATE = '21_12_2018_random/' 

def main():
    env = Ur5(duration=0.01) 
    model = DDPG(a_dim=TRAIN_CONFIG['action_dim'],s_dim=TRAIN_CONFIG['state_dim'])
    model.actor.load_state_dict(torch.load(PATH_TO_MODEL+MODEL_DATE+'actor.pth'))
    model.critic.load_state_dict(torch.load(PATH_TO_MODEL+MODEL_DATE+'critic.pth'))
    
    for epoch in range(TRAIN_CONFIG['test_epoch']): 
        state = env.reset() 
        terminal = False
        for step in range(500):
            action = model.choose_action(state,noise=None)
            state_, reward, terminal = env.step(action*TRAIN_CONFIG['action_bound'])
            state = state_
            if terminal:
                env.reset()
            '''
            if (step+1) % 100 == 0:
                env.target_generate()
                '''

if __name__ == '__main__':
    main()