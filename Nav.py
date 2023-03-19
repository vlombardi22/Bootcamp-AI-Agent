
import torch
import torch.nn as nn

import torch.nn.functional as F

from newutil import set_init




class Net(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        H_1 = 64
        H_2 = 32

        self.actor1 = nn.Linear(s_dim, H_1)
        self.actor2 = nn.Linear(H_1, H_2)
        self.actor3 = nn.Linear(H_2, H_2)
        self.actor4 = nn.Linear(H_2, a_dim)

        self.critic1 = nn.Linear(s_dim, H_1)
        self.critic2 = nn.Linear(H_1, H_2)
        self.critic3 = nn.Linear(H_2, H_2)
        self.critic4 = nn.Linear(H_2, 1)


        set_init([self.actor1, self.actor2, self.actor3, self.actor4, self.critic1, self.critic2, self.critic3, self.critic4])

        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        act1 = F.leaky_relu(self.actor1(x))
        crit1 = F.leaky_relu(self.critic1(x))
        act2 = F.leaky_relu(self.actor2(act1))
        crit2 = F.leaky_relu(self.critic2(crit1))
        act3 = F.leaky_relu(self.actor3(act2))
        crit3 = F.leaky_relu(self.critic3(crit2))
        logits = self.actor4(act3)
        values = self.critic4(crit3)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss
