import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as tdist

#Model input: (n,3) tensor with each column as follows:
#input[:,0]: w_t
#input[:,1]: r_t-alpha_r
#input[:,2]: beta_r

#Model output: (n,1) tensor of epsilon_t
def log_likelihood_update(epsilon,r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda):
    ''' Compute log joint likelihood of l=log p(eps_t,r_t|eps_{t-1},r_{t-1})'''
    # Input:  epsilon  (n,), epsilon_past  (n,), r (1,), r_past (1,)
    # Output: log_prob (n,)

    prior = 0  # do we need prior on r_0, eps_0?

    ''' Calculate u, w, nu and eta'''
    u_past = (r_past - epsilon_past - alpha_r) / beta_r

    w = alpha_u + beta_u * u_past + (gamma + theta * (epsilon_past < 0)) * (epsilon_past ** 2)

    nu = torch.sqrt(beta_r / (r - epsilon - alpha_r)) * epsilon
    nu = torch.where(torch.isnan(nu), torch.tensor(1e10), nu)
    eta = (r - epsilon - alpha_r) / beta_r - w
    eta = torch.where(torch.isnan(eta), torch.tensor(1e6), eta)

    logp_exp = tdist.Normal(0, _lambda).log_prob(eta)
    logp_t = tdist.StudentT(d).log_prob(nu)

    log_joint = logp_exp + logp_t - 0.5 * (torch.log(beta_r) + torch.log(r - epsilon - alpha_r)) + prior

    return log_joint

def param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda):
    #epsilon_past is of shape (n,) everying else scalers
    n=epsilon_past.shape[0]
    output=torch.zeros(n,3)
    u_past=(r_past-alpha_r-epsilon_past)/beta_r
    output[:,0]=alpha_u+beta_u*u_past+gamma*epsilon_past**2+theta*(epsilon_past<0)*epsilon_past**2
    output[:,1]=r-alpha_r
    output[:,2]=beta_r

    return output

class VIScaler(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

learning_rate = 0.001
num_epochs = 1000
batch_size = 20
N=100

model = VIScaler(hidden_size=16)
model.train()

r = torch.randn(N, )
epsilon_past = torch.randn(N, )
r_past = torch.randn(N, )
alpha_r = torch.randn(N, )
beta_r = torch.randn(N, )
d = torch.randn(N, )
alpha_u = torch.randn(N, )
beta_u = torch.randn(N, )
gamma = torch.randn(N, )
theta = torch.randn(N, )
_lambda = torch.randn(N, )

Trainset = param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda)
dataset = TensorDataset(Trainset,r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch in dataloader:
        batch_data = batch[0]
        optimizer.zero_grad()

        outputs = model(batch_data)
        base=torch.distributions.Exponential(1).sample((batch_data.shape[0],))

        logprob=log_likelihood_update(base*outputs,batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8],batch[9],batch[10],batch[11])
        
        loss=torch.sum(-torch.log(outputs)-logprob)
        loss.backward()
        optimizer.step()

