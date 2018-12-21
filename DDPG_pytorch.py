import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

PATH_TO_MODEL = '/home/waiyang/pana_RL_yueci/model_para/'
PATH_TO_PLOT = '/home/waiyang/pana_RL_yueci/model_plot/'

###############################  DDPG  ####################################
def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


class OrnsteinUhlenbeckActionNoise:
    '''Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) to generate 
    temporally corre- lated exploration for exploration efficiency
    '''
    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
	    self.action_dim = action_dim
	    self.mu = mu
	    self.theta = theta
	    self.sigma = sigma
	    self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
	    self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.forward1 = nn.Linear(s_dim, 400)
        self.Relu = nn.ReLU()
        self.forward2 = nn.Linear(400, 300)
        self.forward3 = nn.Linear(300, a_dim)
        self.tanh = nn.Tanh()
        self.ln1 = nn.LayerNorm(400)
        self.ln2 = nn.LayerNorm(300)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = fanin_init(m.weight.data.size())
        self.forward2.weight.data.uniform_(-0.003, 0.003)
        
    def forward(self, x):
        
        x = self.forward1(x)
        #x = self.ln1(x)
        x = self.tanh(x)
        x = self.forward2(x)
        #x = self.ln2(x)
        x = self.Relu(x)
        x = self.forward3(x)
        #x = F.normalize(x)
        x = self.tanh(x)

        return x


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.forward_s = nn.Linear(s_dim, 400)
        self.forward_sa = nn.Linear(400+a_dim, 300)
        self.forward1 = nn.Linear(300, 1)
        self.Relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = fanin_init(m.weight.data.size())
        self.forward1.weight.data.uniform_(-0.003, 0.003)
    
    def forward(self, x, a):
        x = self.forward_s(x)
        x = self.Relu(x)
        x = self.forward_sa(torch.cat([x,a],1))
        x = self.Relu(x)
        x = self.forward1(x)

        return x


class DDPG(object):
    def __init__(
            self,
            a_dim, 
            s_dim, 
            LR_A = 0.0001,    # learning rate for actor 0.001
            LR_C = 0.001,    # learning rate for critic 0.005
            GAMMA = 0.99,     # reward discount  0.9
            TAU = 0.001,      # soft replacement  0.0001
            MEMORY_CAPACITY = 10000,
            BATCH_SIZE = 64,   #32
            cuda = False        
            ):
        self.gama = GAMMA
        self.tau = TAU
        self.memory_size = MEMORY_CAPACITY
        self.batch_size = BATCH_SIZE
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)

        # initialize memory counter
        self.memory_counter = 0
        self.a_dim, self.s_dim = a_dim, s_dim
        self.noise = OrnsteinUhlenbeckActionNoise(self.a_dim)
        self.gpu = cuda

        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim) 
        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)

        self.optim_a = optim.Adam(self.actor.parameters(), LR_A)
        self.optim_c = optim.Adam(self.critic.parameters(), LR_C)
        #self.optim_a = optim.SGD(self.actor.parameters(), LR_A, 0.9)
        #self.optim_c = optim.SGD(self.critic.parameters(), LR_C, 0.9)
        if self.gpu:
            self.cuda = torch.device("cuda")
            self.actor = self.actor.to(self.cuda)
            self.actor_target = self.actor_target.to(self.cuda)
            self.critic = self.critic.to(self.cuda)
            self.critic_target = self.critic_target.to(self.cuda)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        self.loss_actor_list = []
        self.loss_critic_list = []

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def choose_action(self, state, noise=True):
        state = torch.from_numpy(state).float()
        state = state.view(1,-1)
        if self.gpu:
            state = state.to(self.cuda)
            action = self.actor(state.detach())[0]
            action = action.cpu().numpy()
            if noise:
                action += self.noise.sample() 
        else:
            action = self.actor(state.detach())[0]
            action = action.detach().numpy()
            if noise:
                action += self.noise.sample() 
            
        return np.clip(action,-1,1)
    


    def soft_update(self, target, source, tau):
        """
        Copies the parameters from network to target network using the below update
        y = TAU*x + (1 - TAU)*y
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        """
        Copies the parameters from network to target network entirely
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
		    target_param.data.copy_(param.data)

    def Learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_memory = torch.from_numpy(batch_memory)
        if self.gpu:
            batch_memory = torch.from_numpy(batch_memory).to(self.cuda)

        s = batch_memory[:, :self.s_dim].float()
        s_ = batch_memory[:, -self.s_dim:].float()
        a = batch_memory[:, self.s_dim:self.s_dim + self.a_dim].float()
        r = batch_memory[:, self.s_dim + self.a_dim].float()

		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
        a_ = self.actor_target(s_).detach()
        
        next_val = torch.squeeze(self.critic_target(s_, a_).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r + self.gama * next_val
		# y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic(s, a))
		# compute critic loss, and update the critic
		#loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        loss_critic = F.mse_loss(y_predicted, y_expected)
        self.optim_c.zero_grad()
        loss_critic.backward()
        self.optim_c.step()
        self.loss_critic_list.append(loss_critic)
        
		# ---------------------- optimize actor ----------------------
        pred_a = self.actor(s)
        loss_actor = -1 * torch.mean(self.critic(s, pred_a))
        self.optim_a.zero_grad()
        loss_actor.backward()
        #torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1)
        self.optim_a.step()
        self.loss_actor_list.append(loss_actor)

        # ------------------ update target network ------------------
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
       
    def save_model(self, model_name):
        torch.save(self.actor.state_dict(), os.path.join(PATH_TO_MODEL, model_name, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(PATH_TO_MODEL, model_name, 'critic.pth'))
        torch.save(self.optim_a.state_dict(), os.path.join(PATH_TO_MODEL, model_name, 'optim_a.pth'))
        torch.save(self.optim_c.state_dict(), os.path.join(PATH_TO_MODEL, model_name, 'optim_c.pth'))

    def plot_loss(self,model_name):
        plt.figure()
        plt.plot(np.arange(len(self.loss_actor_list)),self.loss_actor_list )
        plt.ylabel('Loss_Actor')
        plt.xlabel('training step')
        plt.savefig(PATH_TO_PLOT+model_name+'loss_actor.png')

        plt.figure()
        plt.plot(np.arange(len(self.loss_critic_list)),self.loss_critic_list )
        plt.ylabel('Loss_Critic')
        plt.xlabel('training step')
        plt.savefig(PATH_TO_PLOT+model_name+'loss_critic.png')




