from env import Ur5
from DDPG import DDPG
from TD3 import TD3
import numpy as np 
import matplotlib.pyplot as plt
import torch
import time
import argparse
import os


def get_time(start_time):
    m, s = divmod(int(time.time()-start_time), 60)
    h, m = divmod(m, 60)
    print 'Total time spent: %d:%02d:%02d' % (h, m, s)

def train(args):

    if not os.path.exists(args.path_to_model+args.model_name+args.model_date):
        os.makedirs(args.path_to_model+args.model_name+args.model_date)

    env = Ur5()

    if args.model_name == 'TD3': model = TD3(a_dim=env.action_dim,s_dim=env.state_dim)
    if args.model_name == 'DDPG': model = DDPG(a_dim=env.action_dim,s_dim=env.state_dim)
    
    #load pre_trained model        
    if args.pre_train:
        model.load_model(args.path_to_model+args.model_name, args.model_date_+'/')
        #load memory data
        memory = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/memory.txt')
        model.memory = memory

    total_reward_list = []
    start_time = time.time()

    for epoch in range(args.train_epoch): 
        state = env.reset()
        total_reward = 0
        for i in range(args.train_step):
            action = model.choose_action(state)
            state_, reward, terminal = env.step(action*args.action_bound)
            model.store_transition(state,action,reward,state_,terminal)
            state = state_
            total_reward += reward
            #start optimization after more than 100 transitions
            if model.memory_counter > 100:
                model.Learn()
            if terminal:
                state = env.reset()
            
        total_reward_list.append(total_reward)
        print 'epoch:', epoch,  '||',  'Reward:', total_reward

        if (epoch+1) % args.epoch_store == 0:
            model.save_model(args.path_to_model+args.model_name, args.model_date+'/')
            #save memory
            memory = model.memory
            np.savetxt(args.path_to_model+args.model_name+args.model_date+'/memory.txt',memory)
            model.plot_loss(args.path_to_model+args.model_name, args.model_date+'/')

    plt.figure()
    plt.plot(np.arange(len(total_reward_list)), total_reward_list)
    plt.ylabel('Total_reward')
    plt.xlabel('training epoch')
    plt.savefig(args.path_to_model+args.model_name+args.model_date+'/reward.png')
    get_time(start_time)

def test(args):
    env = Ur5(duration=0.01)

    if args.model_name == 'TD3': model = TD3(a_dim=env.action_dim,s_dim=env.state_dim)
    if args.model_name == 'DDPG': model = DDPG(a_dim=env.action_dim,s_dim=env.state_dim)
    model.load_model(args.path_to_model+args.model_name, args.model_date_+'/')
   
    for epoch in range(args.test_epoch): 
        state = env.reset()
        total_reward = 0
        for step in range(args.test_step):
            action = model.choose_action(state,noise=None)
            state_, reward, terminal = env.step(action*args.action_bound)
            state = state_
            total_reward += reward
            if terminal:
                env.reset()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='DDPG')
    #Folder name saved as date
    parser.add_argument('--model_date', default='/27_12_2018')
    #Folder stored with trained model weights, which are used for transfer learning
    parser.add_argument('--model_date_', default='/27_12_2018')
    parser.add_argument('--pre_train', default=False)
    parser.add_argument('--path_to_model', default='/home/waiyang/pana_RL_yueci/')
    parser.add_argument('--action_bound', default=np.pi/36, type=float)
    parser.add_argument('--train_epoch', default=400, type=int)
    parser.add_argument('--train_step', default=200, type=int)
    parser.add_argument('--test_epoch', default=100, type=int)
    parser.add_argument('--test_step', default=100, type=int)
    #store the model weights and plot after epoch number
    parser.add_argument('--epoch_store', default=10, type=int)
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    if args.mode == 'train': train(args)
    if args.mode == 'test': test(args)
