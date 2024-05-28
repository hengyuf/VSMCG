import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as tdist
import matplotlib.pyplot as plt

#Model input: (n,3) tensor with each column as follows:
#input[:,0]: w_t
#input[:,1]: r_t-alpha_r
#input[:,2]: beta_r

#Model output: (n,1) tensor of epsilon_t
def log_likelihood_update(epsilon, r, epsilon_past, r_past, alpha_r, beta_r, d, alpha_u, beta_u, gamma, theta, _lambda):
    prior = torch.zeros_like(epsilon)

    u_past = (r_past - epsilon_past - alpha_r) / beta_r

    w = alpha_u + beta_u * u_past + (gamma + theta * (epsilon_past < 0)) * (epsilon_past ** 2)

    nu = torch.sqrt(beta_r / (r - epsilon - alpha_r)) * epsilon
    #nu = torch.where(torch.isnan(nu), torch.tensor(1e10), nu)
    eta = (r - epsilon - alpha_r) / beta_r - w
    #eta = torch.where(torch.isnan(eta), torch.tensor(1e6), eta)

    logp_exp = torch.distributions.Normal(0, 0.5).log_prob(eta).reshape(-1)
    logp_t = torch.distributions.StudentT(d, 0, 1).log_prob(nu).reshape(-1)

    # print("all shape:",epsilon.shape,w.shape,nu.shape,eta.shape,logp_exp.shape,logp_t.shape)

    log_joint = logp_exp + logp_t - 0.5 * (torch.log(beta_r) + torch.log(r - epsilon - alpha_r)) + prior
    #print("log_joint shape:",log_joint.shape)

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
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x=  1e-2+torch.sigmoid(x)
        return x

learning_rate = 1e-3
num_epochs = 1000
batch_size = 256
N=16384

model = VIScaler(hidden_size=16)
model.train()

scale=1
gen_data=False
if gen_data:

    r = torch.rand(N,)*scale-scale/2
    epsilon_past = torch.rand(N,)*scale-scale/2
    r_past = torch.rand(N,)*scale-scale/2
    alpha_r = torch.rand(N,)*scale
    beta_r = torch.rand(N,)*scale
    d = 6*torch.ones(N, )
    alpha_u =torch.rand(N,)*2*scale
    beta_u = torch.rand(N,)*scale
    gamma = torch.rand(N,)*scale
    theta = torch.rand(N,)*scale
    _lambda = torch.rand(N,)*scale

    Trainset = param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda)
    dataset = TensorDataset(Trainset,r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    torch.save(dataloader,"./data.pt")
else:
    dataloader=torch.load("./data.pt")

optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001,)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.5)
loss_epoch=[]
for epoch in range(num_epochs):

    loss_list=[]
    for batch in dataloader:
        batch_data = batch[0]
        optimizer.zero_grad()
        

        outputs = model(batch_data).reshape(-1)
        base=torch.distributions.Exponential(1).sample((batch_data.shape[0],)).reshape(-1)


        logprob=log_likelihood_update(batch[1]-batch[4]-base*outputs,batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8],batch[9],batch[10],batch[11])
        
        loss=torch.mean(-base-torch.log(outputs)-logprob,dim=0)
        #print(logprob.shape) #logq_prob-logp_prob
        if loss.item()<512:
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()
    print(f"----------EPOCH {epoch}------------")
    loss_list=torch.tensor(loss_list)
    print(f"Loss:{torch.mean(loss_list)} Total batch:{loss_list.shape} Loss:{loss.item()}")
    loss_epoch.append(torch.mean(loss_list))
    if epoch%50==49:
        plt.figure()
        plt.xscale("log")
        plt.plot(loss_epoch)
        plt.savefig(f"./figs/Loss_test1_Epoch1-{epoch}.png")
        torch.save(model,f"VIScaler_test1_{epoch}.pth")
    if epoch>100 and loss_epoch[-1]<= min(loss_epoch)+1e-4:
        torch.save(model,f"VIScaler_test1_{epoch}_loss_{loss_epoch[-1]}.pth")
    scheduler.step()



