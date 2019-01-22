from env import Ur5
from env2 import Ur5_vision
from DDPG import DDPG
from TD3 import TD3
from TD3_vision import TD3_vision
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

def img_transform(img_memory, mode, frame_size=4):
    assert type(img_memory).__module__ == np.__name__, 'data type is not numpy'
    assert mode == 'img2txt' or mode == 'txt2img', 'Please use correct mode name 1.img2txt 2.txt2img'

    if mode == 'img2txt':
        size = img_memory.shape[0]
        img_memory = img_memory.reshape((size,-1))
    
    if mode == 'txt2img':
        size = img_memory.shape[0]
        h = w = np.sqrt(img_memory.shape[1] / (2 * frame_size))
        img_memory = img_memory.reshape((size,2,frame_size,h,w))
    
    return img_memory

def train(args, env, model):

    if not os.path.exists(args.path_to_model+args.model_name+args.model_date):
        os.makedirs(args.path_to_model+args.model_name+args.model_date)

    #training reward list
    total_reward_list = np.array([])
    #testing reward and steps list
    test_reward_list, test_step_list = np.array([]), np.array([])
    start_time = time.time()
    if args.pre_train:
        #load pre_trained model 
        try:
            model.load_model(args.path_to_model+args.model_name, args.model_date_+'/')
            print 'load model successfully'
        except:
            print 'fail to load model, check the path of models'

        print 'start random exploration for adding experience'
        if args.model_name == 'TD3_vision':
            vision, state = env.reset()
        else:
            state = env.reset()
        for step in range(args.random_exploration):
            if args.model_name == 'TD3_vision':
                vision_, state_, action, reward, terminal = env.uniform_exploration(np.random.uniform(-1,1,5)*args.action_bound*5)
                model.store_transition(vision,state,action,reward,vision_,state_,terminal)
                state = state_
                vision = vision_
                if terminal:
                    vision, state = env.reset()
            else:
                state_, action, reward, terminal = env.uniform_exploration(np.random.uniform(-1,1,5)*args.action_bound*5)
                model.store_transition(state,action,reward,state_,terminal)
                state = state_
                if terminal:
                    state = env.reset()
        total_reward_list = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/reward.txt')
        test_reward_list = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/test_reward.txt')
        test_step_list = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/test_step.txt')

    print 'start training'
    model.mode(mode='train')

    #training for vision observation
    for epoch in range(args.train_epoch):
        if args.model_name == 'TD3_vision': 
            vision, state = env.reset()
        else:
            state = env.reset()
        total_reward = 0
        for i in range(args.train_step):
            if args.model_name == 'TD3_vision':
                action = model.choose_action(vision,state)
                vision_, state_, reward, terminal = env.step(action*args.action_bound)
                model.store_transition(vision,state,action,reward,vision_,state_,terminal)
                state = state_
                vision = vision_
                total_reward += reward
                if model.memory_counter > args.random_exploration:
                    model.Learn()
                if terminal:
                    vision, state = env.reset()
            else:
                action = model.choose_action(state)
                state_, reward, terminal = env.step(action*args.action_bound)
                model.store_transition(state,action,reward,state_,terminal)
                state = state_
                total_reward += reward
                if model.memory_counter > args.random_exploration:
                    model.Learn()
                if terminal:
                    state = env.reset()
                
        np.append(total_reward_list,total_reward)
        print 'epoch:', epoch,  '||',  'Reward:', total_reward

        #begin testing and record the evalation metrics
        if (epoch+1) % args.test_epoch == 0:
            model.save_model(args.path_to_model+args.model_name, args.model_date+'/')
            model.plot_loss(args.path_to_model+args.model_name, args.model_date+'/')

            avg_reward, avg_step = test(args, env, model)
            model.mode(mode='train')
            print 'finish testing'
            np.append(test_reward_list,avg_reward)
            np.append(test_step_list,avg_step)
            plt.figure()
            plt.plot(np.arange(len(test_reward_list)), test_reward_list)
            plt.ylabel('test_reward')
            plt.xlabel('training epoch / testing epoch')
            plt.savefig(args.path_to_model+args.model_name+args.model_date+'/test_reward.png')
            plt.close()
            np.savetxt(args.path_to_model+args.model_name+args.model_date+'/test_reward.txt',np.array(test_reward_list))

            plt.figure()
            plt.plot(np.arange(len(test_step_list)), test_step_list)
            plt.ylabel('test_step')
            plt.xlabel('training epoch / testing epoch')
            plt.savefig(args.path_to_model+args.model_name+args.model_date+'/test_step.png')
            plt.close()
            np.savetxt(args.path_to_model+args.model_name+args.model_date+'/test_step.txt',np.array(test_step_list))

            plt.figure()
            plt.plot(np.arange(len(total_reward_list)), total_reward_list)
            plt.ylabel('Total_reward')
            plt.xlabel('training epoch')
            plt.savefig(args.path_to_model+args.model_name+args.model_date+'/reward.png')
            plt.close()
            np.savetxt(args.path_to_model+args.model_name+args.model_date+'/reward.txt',np.array(total_reward_list))

            get_time(start_time)    

def test(args, env, model):
    model.mode(mode='test')
    print 'start to test the model'
    try:
        model.load_model(args.path_to_model+args.model_name, args.model_date_+'/')
        print 'load model successfully'
    except:
        print 'fail to load model, check the path of models'

    total_reward_list = []
    steps_list = []
    #testing for vision observation
    for epoch in range(args.test_epoch):
        if args.model_name == 'TD3_vision': 
            vision, state = env.reset()
        else:
            state = env.reset()
        total_reward = 0
        for step in range(args.test_step):
            if args.model_name == 'TD3_vision':
                action = model.choose_action(vision,state,noise=None)
                vision_, state_, reward, terminal = env.step(action*args.action_bound)
                state = state_
                vision = vision_
                total_reward += reward
                if env.get_rotation > env.threshold:
                    steps_list.append(env.steps)
                if terminal:
                    vision, state = env.reset()
            else:
                action = model.choose_action(state,noise=None)
                state_, reward, terminal = env.step(action*args.action_bound)
                state = state_
                total_reward += reward
                if terminal:
                    env.reset()
        total_reward_list.append(total_reward)
        print 'testing_epoch:', epoch,  '||',  'Reward:', total_reward
        
    average_reward = np.mean(np.array(total_reward_list))
    average_step = 0 if steps_list == [] else np.mean(np.array(steps_list))

    return average_reward, average_step


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #select env to be used
    parser.add_argument('--env_name', default='vision')
    #select model to be used
    parser.add_argument('--model_name', default='TD3_vision')
    #Folder name saved as date
    parser.add_argument('--model_date', default='/22_01_2019')
    #Folder stored with trained model weights, which are used for transfer learning
    parser.add_argument('--model_date_', default='/22_01_2019')
    parser.add_argument('--pre_train', default=False)
    parser.add_argument('--path_to_model', default='/home/waiyang/pana_RL_yueci/')
    #The maximum action limit
    parser.add_argument('--action_bound', default=np.pi/72, type=float) #pi/36 for reaching
    parser.add_argument('--train_epoch', default=500, type=int)
    parser.add_argument('--train_step', default=200, type=int)
    parser.add_argument('--test_epoch', default=10, type=int)
    parser.add_argument('--test_step', default=200, type=int)
    #exploration (randome action generation) steps before updating the model
    parser.add_argument('--random_exploration', default=1000, type=int)
    #store the model weights and plots after epoch number
    parser.add_argument('--epoch_store', default=10, type=int)
    #Wether to use GPU
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    assert args.env_name == 'empty' or 'vision', 'env name: 1.empty 2.vision'
    if args.env_name == 'empty': env = Ur5()
    if args.env_name == 'vision': env = Ur5_vision()

    assert args.model_name == 'TD3_vision' or 'TD3' or 'DDPG', 'model name: 1.TD3_vision 2.TD3 3.DDPG'
    if args.model_name == 'TD3_vision': model = TD3_vision(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)
    if args.model_name == 'TD3': model = TD3(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)
    if args.model_name == 'DDPG': model = DDPG(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)

    assert args.mode == 'train' or 'test', 'mode: 1.train 2.test'
    if args.mode == 'train': 
        train(args, env, model)

    if args.mode == 'test': 
        env.duration = 0.1
        test(args, env, model)
