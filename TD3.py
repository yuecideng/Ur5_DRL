import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

###############################  TD3  ####################################
def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


class OrnsteinUhlenbeckActionNoise:
    '''Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) to generate 
    temporally random process for exploration 
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
        self.forward2 = nn.Linear(400, 300)
        self.forward3 = nn.Linear(300, a_dim)
        self.Relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.forward1(x)
        x = self.tanh(x)
        x = self.forward2(x)
        x = self.Relu(x)
        x = self.forward3(x)
        x = self.tanh(x)

        return x


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.forward1 = nn.Linear(s_dim+a_dim, 400)
        self.forward2 = nn.Linear(400, 300)
        self.forward3 = nn.Linear(300, 1)
        self.forward4 = nn.Linear(s_dim+a_dim, 400)
        self.forward5 = nn.Linear(400, 300)
        self.forward6 = nn.Linear(300, 1)
        self.Relu = nn.ReLU()
        
    def forward(self, x, a):
        x1 = self.forward1(torch.cat([x,a],1))
        x1 = self.Relu(x1)
        x1 = self.forward2(x1)
        x1 = self.Relu(x1)
        x1 = self.forward3(x1)

        x2 = self.forward1(torch.cat([x,a],1))
        x2 = self.Relu(x2)
        x2 = self.forward2(x2)
        x2 = self.Relu(x2)
        x2 = self.forward3(x2)

        return x1, x2

    def Q1(self, x, a):
        x1 = self.forward1(torch.cat([x,a],1))
        x1 = self.Relu(x1)
        x1 = self.forward2(x1)
        x1 = self.Relu(x1)
        x1 = self.forward3(x1)

        return x1


class TD3(object):
    def __init__(
            self,
            a_dim, 
            s_dim, 
            LR_A = 0.001,    # learning rate for actor 
            LR_C = 0.001,    # learning rate for critic 
            GAMMA = 0.99,     # reward discount  
            TAU = 0.005,      # soft replacement 
            MEMORY_CAPACITY = 100000,
            BATCH_SIZE = 100,   #32
            act_noise = 0.1,
            target_noise = 0.2,
            noise_clip = 0.5,
            policy_delay = 2,
            cuda = False       
            ):
        self.gama = GAMMA
        self.tau = TAU
        self.memory_size = MEMORY_CAPACITY
        self.batch_size = BATCH_SIZE
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        #memory to store the [state,action,reward,next_state,done] transition
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
        # initialize memory counter
        self.memory_counter = 0
        #state and action dimension
        self.a_dim, self.s_dim = a_dim, s_dim
        self.noise = OrnsteinUhlenbeckActionNoise(self.a_dim)
        #set target noise with small sigma
        self.noise_target = OrnsteinUhlenbeckActionNoise(self.a_dim,sigma=0.1)
        self.gpu = cuda

        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim) 
        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)

        self.optim_a = optim.Adam(self.actor.parameters(), LR_A)
        self.optim_c = optim.Adam(self.critic.parameters(), LR_C)
    
        if self.gpu:
            self.cuda = torch.device("cuda")
            self.actor = self.actor.to(self.cuda)
            self.actor_target = self.actor_target.to(self.cuda)
            self.critic = self.critic.to(self.cuda)
            self.critic_target = self.critic_target.to(self.cuda)
 
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        self.loss_actor_list = []
        self.critic1_q = []
        self.critic2_q = []

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.memory_counter % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def choose_action(self, state, noise=True):
        state = torch.from_numpy(state).float()
        state = state.view(1,-1)
        if self.gpu:
            state = state.to(self.cuda)
            action = self.actor(state).flatten()
            action = action.cpu().detach().numpy()
            if noise:
                #action += np.random.normal(0,self.act_noise,self.a_dim)
                action += self.noise.sample()
        else:
            action = self.actor(state).flatten()
            action = action.detach().numpy()
            if noise:
                #add Guassian noise
                #action += np.random.normal(0,self.act_noise,self.a_dim)
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
        batch_memory = torch.from_numpy(batch_memory).float()
        if self.gpu:
            batch_memory = batch_memory.to(self.cuda)

        s = batch_memory[:, :self.s_dim]
        s_ = batch_memory[:, -self.s_dim-1:-1]
        a = batch_memory[:, self.s_dim:self.s_dim + self.a_dim]
        r = batch_memory[:, self.s_dim + self.a_dim].view(-1,1)
        d = batch_memory[:, -1].view(-1,1)
		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
        a_ = self.actor_target(s_) + torch.clamp(torch.from_numpy(self.noise_target.sample()).float(),
                                                            min=-self.noise_clip,max=self.noise_clip)
        a_ = torch.clamp(a_,min=-1,max=1).detach()
        #a_ = self.actor_target(s_)
        q1, q2 = self.critic_target(s_, a_)
        q_ = torch.min(q1,q2)
		# y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r + (self.gama * (1-d) * q_).detach()
		# y_pred = Q( s1, a1)
        y_predicted1, y_predicted2 = self.critic(s, a)
		# compute critic loss, and update the critic
        loss_critic = F.mse_loss(y_predicted1, y_expected) + F.mse_loss(y_predicted2, y_expected)
        self.optim_c.zero_grad()
        loss_critic.backward()
        self.optim_c.step()
      
        self.critic1_q.append(torch.mean(y_predicted1))
        self.critic2_q.append(torch.mean(y_predicted2))
        
		# ---------------------- optimize actor ----------------------
        if self.memory_counter % self.policy_delay == 0:
            pred_a = self.actor(s)
            loss_actor = -self.critic.Q1(s, pred_a).mean()
            self.optim_a.zero_grad()
            loss_actor.backward()
            self.optim_a.step()
            self.loss_actor_list.append(loss_actor)

            # ------------------ update target network ------------------
            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target, self.critic, self.tau)
       
    def save_model(self,model_dir,model_name):
        torch.save(self.actor.state_dict(), model_dir+model_name+'actor.pth')
        torch.save(self.critic.state_dict(), model_dir+model_name+'critic.pth')
        torch.save(self.optim_a.state_dict(), model_dir+model_name+'optim_a.pth')
        torch.save(self.optim_c.state_dict(), model_dir+model_name+'optim_c.pth')

    def load_model(self,model_dir,model_name):
        self.actor_target.load_state_dict(torch.load(model_dir+model_name+'actor.pth'))
        self.critic_target.load_state_dict(torch.load(model_dir+model_name+'critic.pth'))
        self.actor.load_state_dict(torch.load(model_dir+model_name+'actor.pth'))
        self.critic.load_state_dict(torch.load(model_dir+model_name+'critic.pth'))
        self.optim_a.load_state_dict(torch.load(model_dir+model_name+'optim_a.pth'))
        self.optim_c.load_state_dict(torch.load(model_dir+model_name+'optim_c.pth'))

    def plot_loss(self,model_dir,model_name):
        plt.figure()
        plt.plot(np.arange(len(self.loss_actor_list)),self.loss_actor_list )
        plt.ylabel('Loss_Actor')
        plt.xlabel('training step')
        plt.savefig(model_dir+model_name+'loss_actor.png')
        plt.close()

        plt.figure()
        plt.plot(np.arange(len(self.critic1_q)),self.critic1_q)
        plt.ylabel('Q value')
        plt.xlabel('training step')
        plt.savefig(model_dir+model_name+'Q_critic1.png')
        plt.close()

        plt.figure()
        plt.plot(np.arange(len(self.critic2_q)),self.critic2_q)
        plt.ylabel('Q value')
        plt.xlabel('training step')
        plt.savefig(model_dir+model_name+'Q_critic2.png')
        plt.close()

    def mode(self, mode='train'):
        if mode == 'train':
            self.actor.train()
            self.critic.train()
        if mode == 'test':
            self.actor.eval()
            self.critic.eval()