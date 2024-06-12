import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as tdist
import matplotlib.pyplot as plt
import random
from scipy.stats import t, expon, norm
from TruncatedNormal import TruncatedNormal
import numpy as np
#from generate_train_set import generate_train
from torch.distributions import Normal

from tqdm import tqdm, trange

class HalfNormal:
    def __init__(self, scale):
        self.normal = Normal(0, scale)
    
    def sample(self, sample_shape=torch.Size()):
        return torch.abs(self.normal.sample(sample_shape))
    
    def log_prob(self, value):
        # Only defined for value >= 0
        if torch.any(value < 0):
            return torch.tensor(float('-inf'))
        return self.normal.log_prob(value) + torch.log(torch.tensor(2.0))
    



base_dist = tdist.Gamma(1.5,0.75)


gen_data=False

#Model input: (n,3) tensor with each column as follows:
#input[:,0]: w_t
#input[:,1]: r_t-alpha_r
#input[:,2]: beta_r

#Model output: (n,1) tensor of epsilon_t
def log_likelihood_update(epsilon, r, epsilon_past, r_past, alpha_r, beta_r, d, alpha_u, beta_u, gamma, theta, _lambda):
    prior = torch.zeros_like(epsilon)

    u_past = (r_past - epsilon_past - alpha_r) / beta_r

    w = alpha_u + beta_u * u_past + (gamma + theta * (epsilon_past < 0)) * (epsilon_past ** 2)

    ww=r - epsilon - alpha_r+1e-6

    #print("modified base2",r - epsilon - alpha_r)
    if ww.min()<0:

        print("min Error occuered, counts:", (ww<0).sum())
        ww=ww*(ww>=0)+1e-6

    # print("ww:",ww.min(),ww.max())
    #print(f"beta_r:{beta_r} eps:{epsilon.max()}")

    nu = torch.sqrt(beta_r / (ww)) * epsilon
    # print("nu:",nu.min(),nu.max())
    #nu = torch.where(torch.isnan(nu), torch.tensor(1e10), nu)
    eta = (ww) / beta_r - w
    # print("eta:",eta.min(),eta.max())
    #eta = torch.where(torch.isnan(eta), torch.tensor(1e6), eta)

    logp_exp = torch.distributions.Normal(0, 0.5).log_prob(eta).reshape(-1)
    logp_t = torch.distributions.StudentT(d, 0, 1).log_prob(nu).reshape(-1)

    # print("all shape:",epsilon.shape,w.shape,nu.shape,eta.shape,logp_exp.shape,logp_t.shape)

    log_joint = logp_exp + logp_t - 0.5 * (torch.log(beta_r) + torch.log(ww)) + prior
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
        self.fc12= nn.Linear(hidden_size, hidden_size)
        self.fc13= nn.Linear(hidden_size, hidden_size)


        self.fc21 = nn.Linear(hidden_size, 1)
        self.fc22 = nn.Linear(hidden_size, 1)
        self.fc23 = nn.Linear(hidden_size, 1)
        self.fc24 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc12(x))
        x= torch.relu(self.fc13(x))
        x1 = self.fc21(x)
        x2 = self.fc22(x)
        x3 = self.fc23(x)
        x4 = self.fc24(x)
        x1 = 1e-2+torch.sigmoid(x1)
        x2 = 1e-4+torch.sigmoid(x2)
        x3 = 1e-4+torch.sigmoid(x3)
        x4 = 1e-4+torch.sigmoid(x4)
        return x1,x2,x3,x4




def gen_data(N=100,T=100,scale=1):

    scale=1
    alpha_r_list=torch.zeros((N,T))
    beta_r_list=torch.zeros((N,T))
    d_list=torch.zeros((N,T))
    alpha_u_list=torch.zeros((N,T))
    beta_u_list=torch.zeros((N,T))
    gamma_list=torch.zeros((N,T))
    theta_list=torch.zeros((N,T))
    _lambda_list=torch.zeros((N,T))
    eps_list=torch.zeros((N,T))
    eps_past_list=torch.zeros((N,T))
    r_list=torch.zeros((N,T))
    r_past_list=torch.zeros((N,T))

    for i in range(N):


        alpha_r = (torch.rand(1)*4*scale-2*scale).item()   #0*torch.ones(N, ) #torch.rand(N,)*scale#
        beta_r = (torch.rand(1)*scale*1).item()#0.5*torch.ones(N, ) #torch.rand(N,)*scale
        d = 6
        alpha_u =  (torch.rand(1)*4*scale).item()+1#0.5*torch.ones(N, ) #torch.rand(N,)*2*scale
        beta_u = (torch.rand(1)*scale*0.4).item()#0.2*torch.ones(N, )   #torch.rand(N,)*scale
        gamma = (torch.rand(1)*0.4*scale).item()#0*torch.ones(N, )   #torch.rand(N,)*scale
        theta = (torch.rand(1)*scale*0.4).item()#0*torch.ones(N, )   #torch.rand(N,)*scale
        _lambda = (torch.rand(1)*scale).item()#4*torch.ones(N, ) #torch.rand(N,)*scale

        if i<=10000:
            #alpha_r,beta_r,d,alpha_u,beta_u,gamma,theta,_lambda=0.2+np.random.randn()*0.05, 0.2+np.random.randn()*0.05, 6.0+np.random.randn(), 1+np.random.randn()*0.2, 0.4+np.random.randn()*0.1, 0.1+np.random.randn()*0.02, 0.02+np.random.randn()*0.002, 2.5
            alpha_r,beta_r,d,alpha_u,beta_u,gamma,theta,_lambda=0.2, 0.1+np.random.uniform()*0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5

        eps=0
        u=0
        eps_templist=np.zeros(T+1)
        r_templist=np.zeros(T+1)
        for t in range(T+1):
            u = alpha_u+beta_u*u+gamma*eps**2+theta*(eps<0)*eps*eps+norm.rvs(scale=0.5)
            eps=np.random.standard_t(df=d)*np.sqrt(u)
            #print(u,eps)
            r_templist[t]=alpha_r+beta_r*u+eps
            eps_templist[t]=eps

        while np.isnan(eps_templist).sum()+np.isnan(r_templist).sum()>0 or np.abs(r_templist).max()>100:
            print(f"step {i} resample")
            # alpha_r = (torch.rand(1)*10*scale-5*scale).item()   #0*torch.ones(N, ) #torch.rand(N,)*scale#
            # beta_r = (torch.rand(1)*scale).item()#0.5*torch.ones(N, ) #torch.rand(N,)*scale
            # d = 6
            # alpha_u =  (torch.rand(1)*10*scale).item()+2#0.5*torch.ones(N, ) #torch.rand(N,)*2*scale
            # beta_u = (torch.rand(1)*scale*0.2).item()#0.2*torch.ones(N, )   #torch.rand(N,)*scale
            # gamma = (torch.rand(1)*2*scale-scale).item()#0*torch.ones(N, )   #torch.rand(N,)*scale
            # theta = (torch.rand(1)*2*scale-scale).item()#0*torch.ones(N, )   #torch.rand(N,)*scale
            # _lambda = (torch.rand(1)*scale).item()#4*torch.ones(N, ) #torch.rand(N,)*scale

            alpha_r,beta_r,d,alpha_u,beta_u,gamma,theta,_lambda=0.2, 0.1+np.random.uniform()*0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5
            eps=0
            u=0
            eps_templist=np.zeros(T+1)
            r_templist=np.zeros(T+1)
            for t in range(T+1):
                if eps<0:
                    u = alpha_u+beta_u*u+gamma*eps**2+theta*eps*eps+norm.rvs(scale=0.5)
                else:
                    u = alpha_u+beta_u*u+gamma*eps**2+norm.rvs(scale=0.5)
                eps=np.random.standard_t(df=d)*np.sqrt(u)
                #print(u,eps)
                r_templist[t]=alpha_r+beta_r*u+eps
                eps_templist[t]=eps
        #print("rmax:",r_templist.max())


        eps_templist=torch.tensor(eps_templist)
        r_templist=torch.tensor(r_templist)
        alpha_r_list[i]=alpha_r*torch.ones(T)
        beta_r_list[i]=torch.ones(T)*beta_r
        d_list[i]=torch.ones(T)*d
        alpha_u_list[i]=torch.ones(T)*alpha_u
        beta_u_list[i]=torch.ones(T)*beta_u
        gamma_list[i]=torch.ones(T)*gamma
        theta_list[i]=torch.ones(T)*theta
        _lambda_list[i]=torch.ones(T)*_lambda
        eps_list[i]=eps_templist[1:]
        eps_past_list[i]=eps_templist[:T]
        r_list[i]=r_templist[1:]
        r_past_list[i]=r_templist[:T]
    
    
    return r_list.reshape(-1),eps_past_list.reshape(-1),r_past_list.reshape(-1),\
alpha_r_list.reshape(-1), beta_r_list.reshape(-1), d_list.reshape(-1), alpha_u_list.reshape(-1),\
        beta_u_list.reshape(-1),gamma_list.reshape(-1), theta_list.reshape(-1), _lambda_list.reshape(-1)


scale=1
gen_data_aug=False
N=256
T=1000

learning_rate = 3*1e-3
num_epochs = 200
batch_size = 2560
hidden_size=64
loss_tolerance=10000 #Gradually decay to 0.5*tolerance


model = VIScaler(hidden_size=hidden_size)
model.train()





r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda=gen_data(N=N,T=T,scale=scale)
print("-----------Train Dataset------------")
print("r",r.max())
Trainset = param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda)
#print(Trainset)
dataset = TensorDataset(Trainset,r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("-----------Test Dataset ------------")

alpha_r=torch.ones((N))*0.2
beta_r=torch.ones((N))*0.2
d=torch.ones((N))*6.0
alpha_u=torch.ones((N))*1
beta_u=torch.ones((N))*0.4
gamma=torch.ones((N))*0.1
theta=torch.ones((N))*0.02
_lambda=torch.ones((N))*2.5


eps=torch.randn(N)
epsilon_past=torch.zeros((N))
epsilon_past[1:]=eps[:N-1]
r=torch.randn(N)
r_past=torch.zeros((N))
r_past[1:]=r[:N-1]

Testset = param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda)
#print(Trainset)
test_dataset = TensorDataset(Testset,r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda)
test_dataloader = DataLoader(test_dataset, batch_size=N, shuffle=True)







optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001,)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.7)
loss_epoch=[]


with trange(num_epochs) as t:

    for epoch in t:

        loss_list=[]
        for batch in dataloader:
            batch_data = batch[0]
            optimizer.zero_grad()
            

            outputs,outputs2,outputs3,outputs4 = model(batch_data)
            outputs,outputs2,outputs3,outputs4 =outputs.reshape(-1),outputs2.reshape(-1),outputs3.reshape(-1),outputs4.reshape(-1)
            base_dist=TruncatedNormal(loc=outputs,scale=outputs2,a=1e-4,b=100)

            #print(batch[1].shape)

            modifiedbase= base_dist.rsample(sample_shape=torch.ones(1).shape).reshape(-1)
            #print(modifiedbase.shape)



            
            # base1=base_dist.sample((batch_data.shape[0],)).reshape(-1)
            # #base2=base_dist2.sample((batch_data.shape[0],)).reshape(-1)
            # random_tensor = torch.rand_like(outputs4)

            # base = base1#torch.where(random_tensor < outputs4, base1, base2)

            # modifiedbase=outputs*base+outputs2*base**1.5+outputs3*base**0.5
            # assert modifiedbase.min()>0
            # jacobian=outputs+1.5*base**0.5*outputs2+0.5*outputs3*base**(-0.5)

            # #modifiedbase=outputs*base
            # #jacobian=outputs
            # #print("modified base1",modifiedbase)
            logprob=log_likelihood_update(batch[1]-batch[4]-modifiedbase,batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8],batch[9],batch[10],batch[11])

            # print("modified base:",modifiedbase.min(),modifiedbase.max())
            # print("logprob:",logprob.min(),logprob.max())
            # print(f"batch:{batch[1].max()} {batch[1].min()} {batch[4].max()} {batch[4].min()}")
            
            baselogprob=base_dist.log_prob(modifiedbase)  #torch.log(outputs4*torch.exp(base_dist.log_prob(base))+(1-outputs4)*torch.exp(base_dist2.log_prob(base)))

            loss=torch.mean(baselogprob-logprob,dim=0)#torch.mean(baselogprob-torch.log(jacobian)-logprob,dim=0)


            #print(logprob.shape) #logq_prob-logp_prob
            if loss.item()<loss_tolerance*(0.5+5/(epoch+10)):
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()
            elif random.random()<0.1:
                loss_list.append(loss.item())
                loss=loss*0.2
                loss.backward()
                optimizer.step()

        for batch in test_dataloader:
            batch_data = batch[0]

            outputs,outputs2,outputs3,outputs4 = model(batch_data)
            outputs,outputs2,outputs3,outputs4 =outputs.reshape(-1),outputs2.reshape(-1),outputs3.reshape(-1),outputs4.reshape(-1)
            base_dist=TruncatedNormal(loc=outputs,scale=outputs2,a=1e-4,b=100)
            modifiedbase= base_dist.rsample(sample_shape=torch.ones(1).shape).reshape(-1)

            logprob=log_likelihood_update(batch[1]-batch[4]-modifiedbase,batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8],batch[9],batch[10],batch[11])

            
            baselogprob=base_dist.log_prob(modifiedbase)  #torch.log(outputs4*torch.exp(base_dist.log_prob(base))+(1-outputs4)*torch.exp(base_dist2.log_prob(base)))

            test_loss=torch.mean(baselogprob-logprob,dim=0)#torch.mean(baselogprob-torch.log(jacobian)-logprob,dim=0)


        
        loss_list=torch.tensor(loss_list)
        #print(f"Loss:{torch.mean(loss_list)} Total batch:{loss_list.shape} Loss:{loss.item()}")
        loss_epoch.append(torch.mean(loss_list))
        t.set_description(f"Epoch: {epoch}  Train Loss:{torch.mean(loss_list)} Total batch:{loss_list.shape} Test Loss:{test_loss}")
        if epoch%10==9:
            plt.figure()
            plt.yscale("log")
            plt.xscale("log")
            plt.plot(loss_epoch)
            plt.savefig(f"./figs/Loss_test1_Epoch1-{epoch}.png")
            torch.save(model,f"VIScaler_test1_{epoch}.pth")
        if epoch>50 and loss_epoch[-1]<= min(loss_epoch)+1e-4:
            torch.save(model,f"VIScaler_test1_{epoch}_loss_{loss_epoch[-1]}.pth")
        scheduler.step()



