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


class Policy(nn.Module):

    def __init__(self, hidden_size, state_dim, action_dim):
        super(Policy, self).__init__()
        num_outputs = action_dim

        self.bn0 = nn.BatchNorm1d(state_dim)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, num_outputs ** 2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

        self.tril_mask = torch.tril(torch.ones(
            num_outputs, num_outputs), diagonal=-1).unsqueeze(0)
        self.diag_mask = torch.diag(torch.diag(
            torch.ones(num_outputs, num_outputs))).unsqueeze(0)

    def forward(self, inputs):
        x, u = inputs
        x = self.bn0(x)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))

        V = self.V(x)
        mu = torch.tanh(self.mu(x))

        Q = None
        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * \
                self.tril_mask.expand_as(
                    L) + torch.exp(L) * self.diag_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (u - mu).unsqueeze(2)
            A = -0.5 * \
                torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V


class DQN_NAF(object):
    def __init__(
            self,
            a_dim, 
            s_dim, 
            LR = 0.001,    # learning rate for actor 0.001
            GAMMA = 0.9,     # reward discount  0.9
            TAU = 0.001,      # soft replacement  0.0001
            hidden_size = 256,
            MEMORY_CAPACITY = 10000,
            BATCH_SIZE = 64,   #32      
            ):
        self.gama = GAMMA
        self.tau = TAU
        self.memory_size = MEMORY_CAPACITY
        self.batch_size = BATCH_SIZE
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.a_dim, self.s_dim = a_dim, s_dim
        # initialize memory counter
        self.memory_counter = 0
        self.noise = OrnsteinUhlenbeckActionNoise(self.a_dim)

        self.agent = Policy(hidden_size,s_dim,a_dim)
        self.agent_target = Policy(hidden_size,s_dim,a_dim)
        self.optim = optim.Adam(self.agent.parameters(), LR)
        #self.optim_a = optim.SGD(self.actor.parameters(), LR_A, 0.9)
        #self.optim_c = optim.SGD(self.critic.parameters(), LR_C, 0.9)
        self.hard_update(self.agent_target, self.agent)
        self.loss_agent_list = []

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def choose_action(self, state, noise=True):
        state = torch.from_numpy(state).float()
        state = state.view(1,-1)
        self.agent.eval()
        action, _, _ = self.agent((state.detach(),None))
        if noise:
            action = action.detach().numpy()[0] + self.noise.sample() 
            
        return np.clip(action,-1,1)[0]

    def soft_update(self, target, source, tau):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        """
        Copies the parameters from source network to target network
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

        s = batch_memory[:, :self.s_dim].float()
        s_ = batch_memory[:, -self.s_dim:].float()
        a = batch_memory[:, self.s_dim:self.s_dim + self.a_dim].float()
        r = batch_memory[:, self.s_dim + self.a_dim].float()

        state_value_ = self.agent_target((s_,None))[2]
        q_pred = r + self.gama * state_value_
        q = self.agent((s,a))[1]

        loss = F.mse_loss(q_pred, q)
        self.optim.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optim.step()
        # ------------------ update target network ------------------
        self.soft_update(self.agent_target, self.agent, self.tau)
        self.loss_agent_list.append(loss)
       
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




