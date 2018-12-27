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
        self.Relu = nn.ReLU()
        self.forward2 = nn.Linear(400, 300)
        self.forward3 = nn.Linear(300, a_dim)
        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = fanin_init(m.weight.data.size())
        self.forward2.weight.data.uniform_(-0.003, 0.003)
        
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
        self.forward_sa = nn.Linear(s_dim+a_dim, 400)
        self.forward1 = nn.Linear(400, 300)
        self.forward2 = nn.Linear(300, 1)
        self.Relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = fanin_init(m.weight.data.size())
        self.forward1.weight.data.uniform_(-0.003, 0.003)
    
    def forward(self, x, a):
        x = self.forward_sa(torch.cat([x,a],1))
        x = self.Relu(x)
        x = self.forward1(x)
        x = self.Relu(x)
        x = self.forward2(x)

        return x


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
            BATCH_SIZE = 64,   #32
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
        self.gpu = cuda

        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim) 
        self.critic1 = Critic(s_dim, a_dim)
        self.critic1_target = Critic(s_dim, a_dim)
        self.critic2 = Critic(s_dim, a_dim)
        self.critic2_target = Critic(s_dim, a_dim)

        
        self.optim_a = optim.Adam(self.actor.parameters(), LR_A)
        self.optim_c1 = optim.Adam(self.critic1.parameters(), LR_C)
        self.optim_c2 = optim.Adam(self.critic2.parameters(), LR_C)
    
        if self.gpu:
            self.cuda = torch.device("cuda")
            self.actor = self.actor.to(self.cuda)
            self.actor_target = self.actor_target.to(self.cuda)
            self.critic1 = self.critic1.to(self.cuda)
            self.critic1_target = self.critic1_target.to(self.cuda)
            self.critic2 = self.critic2.to(self.cuda)
            self.critic2_target = self.critic2_target.to(self.cuda)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic1_target, self.critic1)
        self.hard_update(self.critic2_target, self.critic2)
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
            action = self.actor(state.detach())[0]
            action = action.cpu().numpy()
            if noise:
                #add Guassian noise
                action += torch.FloatTensor(action).normal_(0, self.act_noise)
                #action += self.noise.sample()
        else:
            action = self.actor(state.detach())[0]
            action = action.detach().numpy()
            if noise:
                action += torch.FloatTensor(action).normal_(0, self.act_noise)
                #action += self.noise.sample()
            
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
        s_ = batch_memory[:, -self.s_dim-1:-1].float()
        a = batch_memory[:, self.s_dim:self.s_dim + self.a_dim].float()
        r = batch_memory[:, self.s_dim + self.a_dim].float()
        d = batch_memory[:, -1]
		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
        a_ = self.actor_target(s_).detach() + torch.clamp(torch.FloatTensor(a).normal_(0,self.target_noise),min=-self.noise_clip,max=self.noise_clip)
        a_ = torch.clamp(a_,min=-1,max=1)

        q1 = torch.squeeze(self.critic1_target(s_, a_).detach())
        q2 = torch.squeeze(self.critic2_target(s_, a_).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r + self.gama * (1 - d) * torch.min(q1,q2)
		# y_pred = Q( s1, a1)
        y_predicted1 = torch.squeeze(self.critic1(s, a))
        y_predicted2 = torch.squeeze(self.critic2(s, a))
		# compute critic loss, and update the critic
		#loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        loss_critic1 = F.mse_loss(y_predicted1, y_expected)
        loss_critic2 = F.mse_loss(y_predicted2, y_expected)
        self.optim_c1.zero_grad()
        self.optim_c2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        self.optim_c1.step()
        self.optim_c2.step()

        self.critic1_q.append(torch.mean(y_predicted1))
        self.critic2_q.append(torch.mean(y_predicted2))
        
		# ---------------------- optimize actor ----------------------
        if self.memory_counter % self.policy_delay == 0:
            pred_a = self.actor(s)
            loss_actor = -1 * torch.mean(self.critic1(s, pred_a))
            self.optim_a.zero_grad()
            loss_actor.backward()
            self.optim_a.step()
            self.loss_actor_list.append(loss_actor)

            # ------------------ update target network ------------------
            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic1_target, self.critic1, self.tau)
            self.soft_update(self.critic2_target, self.critic2, self.tau)
       
    def save_model(self,model_dir,model_name):
        torch.save(self.actor.state_dict(), model_dir+model_name+'actor.pth')
        torch.save(self.critic1.state_dict(), model_dir+model_name+'critic1.pth')
        torch.save(self.critic2.state_dict(), model_dir+model_name+'critic2.pth')
        torch.save(self.optim_a.state_dict(), model_dir+model_name+'optim_a.pth')
        torch.save(self.optim_c1.state_dict(), model_dir+model_name+'optim_c1.pth')
        torch.save(self.optim_c2.state_dict(), model_dir+model_name+'optim_c2.pth')

    def load_model(self,model_dir,model_name):
        self.actor_target.load_state_dict(torch.load(model_dir+model_name+'actor.pth'))
        self.critic1_target.load_state_dict(torch.load(model_dir+model_name+'critic1.pth'))
        self.critic2_target.load_state_dict(torch.load(model_dir+model_name+'critic2.pth'))
        self.actor.load_state_dict(torch.load(model_dir+model_name+'actor.pth'))
        self.critic1.load_state_dict(torch.load(model_dir+model_name+'critic1.pth'))
        self.critic2.load_state_dict(torch.load(model_dir+model_name+'critic2.pth'))
        self.optim_a.load_state_dict(torch.load(model_dir+model_name+'optim_a.pth'))
        self.optim_c1.load_state_dict(torch.load(model_dir+model_name+'optim_c1.pth'))
        self.optim_c2.load_state_dict(torch.load(model_dir+model_name+'optim_c2.pth'))

    def plot_loss(self,model_dir,model_name):
        plt.figure()
        plt.plot(np.arange(len(self.loss_actor_list)),self.loss_actor_list )
        plt.ylabel('Loss_Actor')
        plt.xlabel('training step')
        plt.savefig(model_dir+model_name+'loss_actor.png')

        plt.figure()
        plt.plot(np.arange(len(self.critic1_q)),self.critic1_q)
        plt.ylabel('Q value')
        plt.xlabel('training step')
        plt.savefig(model_dir+model_name+'Q_critic1.png')

        plt.figure()
        plt.plot(np.arange(len(self.critic2_q)),self.critic2_q)
        plt.ylabel('Q value')
        plt.xlabel('training step')
        plt.savefig(model_dir+model_name+'Q_critic2.png')




