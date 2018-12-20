from env import Ur5
from DDPG_pytorch import DDPG
from DQN_NAF import DQN_NAF
import numpy as np 
import matplotlib.pyplot as plt
from math import pi
import torch
import time

#Traning hyperparameters 
TRAIN_CONFIG = {'state_dim':13,'action_dim':5,'train_epoch':500,'train_step':200,
                'pre_trained':False,'cuda':False}
#Model name saved as date
MODEL_DATE = '20_12_2018/'
MODEL_DATE_ = '20_12_2018/'
PATH_TO_PLOT = '/home/waiyang/pana_RL_yueci/model_plot/'
PATH_TO_MODEL = '/home/waiyang/pana_RL_yueci/model_para/'

def load_model(model):
    model.actor_target.load_state_dict(torch.load(PATH_TO_MODEL+MODEL_DATE_+'actor.pth'))
    model.critic_target.load_state_dict(torch.load(PATH_TO_MODEL+MODEL_DATE_+'critic.pth'))
    model.actor.load_state_dict(torch.load(PATH_TO_MODEL+MODEL_DATE_+'actor.pth'))
    model.critic.load_state_dict(torch.load(PATH_TO_MODEL+MODEL_DATE_+'critic.pth'))
    model.optim_a.load_state_dict(torch.load(PATH_TO_MODEL+MODEL_DATE+'optim_a.pth'))
    model.optim_c.load_state_dict(torch.load(PATH_TO_MODEL+MODEL_DATE+'optim_c.pth'))

def get_time(start_time):
    m, s = divmod(int(time.time()-start_time), 60)
    h, m = divmod(m, 60)
    print 'Total time spent: %d:%02d:%02d' % (h, m, s)

def main():
    env = Ur5()
    model = DDPG(a_dim=TRAIN_CONFIG['action_dim'],s_dim=TRAIN_CONFIG['state_dim'])
    #model = DQN_NAF(a_dim=TRAIN_CONFIG['action_dim'],s_dim=TRAIN_CONFIG['state_dim'])
    #load pre_trained model        
    if TRAIN_CONFIG['pre_trained']:
        load_model(model)
        #model.load_model(MODEL_DATE_)
        #load noise data
        noise = np.loadtxt('/home/waiyang/pana_RL_yueci/noise.txt')
        model.noise.X = noise
        memory = np.loadtxt('/home/waiyang/pana_RL_yueci/memory.txt')
        model.memory = memory

    total_reward_list = []
    start_time = time.time()

    for epoch in range(TRAIN_CONFIG['train_epoch']): 
        state = env.reset()
        total_reward = 0
        for i in range(TRAIN_CONFIG['train_step']):
            action = model.choose_action(state)
            state_, reward, terminal = env.step(action)
            model.store_transition(state,action,reward,state_)
            state = state_
            total_reward += reward
            if model.memory_counter > 1000:
                model.Learn()
            if terminal:
                state = env.reset()
            
        total_reward_list.append(total_reward)
        print 'epoch:', epoch,  '||',  'Reward:', total_reward  
        if (epoch+1) % 10 == 0:
            model.save_model(MODEL_DATE)
            #save noise data
            noise = model.noise.X
            np.savetxt('/home/waiyang/pana_RL_yueci/noise.txt',noise)
            #save memory
            memory = model.memory
            np.savetxt('/home/waiyang/pana_RL_yueci/memory.txt',memory)
            model.plot_loss(MODEL_DATE)
    plt.figure()
    plt.plot(np.arange(len(total_reward_list)), total_reward_list)
    plt.ylabel('Total_reward')
    plt.xlabel('training epoch')
    plt.savefig(PATH_TO_PLOT+MODEL_DATE+'reward.png')
    get_time(start_time)

if __name__ == '__main__':
    main()
