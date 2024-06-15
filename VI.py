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
from VIScaler import VIScaler

from tqdm import tqdm, trange

device='cpu'

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
    output[:,0]=alpha_u+beta_u*u_past+gamma*epsilon_past**2+theta*(epsilon_past<0)*epsilon_past**2 #w
    output[:,1]=r-alpha_r 
    output[:,2]=beta_r

    return output


NOISE_SCALE=np.array([5,0.02,0,0,0.1,0.,0.,0])
def gen_finetune_data(N=100,T=100,params=(0, 0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5),noise_scale=NOISE_SCALE,norm_scale=1):
    alpha_r_list=torch.zeros((N,T))
    beta_r_list=torch.zeros((N,T))
    d_list=torch.zeros((N,T))
    alpha_u_list=torch.zeros((N,T))
    beta_u_list=torch.zeros((N,T))
    gamma_list=torch.zeros((N,T))
    theta_list=torch.zeros((N,T))
    _lambda_list=torch.zeros((N,T))

    alpha_r=(params[0]+torch.rand(N)*noise_scale[0]-0.5*noise_scale[0]).reshape(-1,1)
    beta_r=(params[1]+torch.rand(N)*noise_scale[1]-0.5*noise_scale[1]).reshape(-1,1)
    d=(params[2]+torch.rand(N)*noise_scale[2]-0.5*noise_scale[2]).reshape(-1,1)
    alpha_u=(params[3]+torch.rand(N)*noise_scale[3]-0.5*noise_scale[3]).reshape(-1,1)
    beta_u=(params[4]+torch.rand(N)*noise_scale[4]-0.5*noise_scale[4]).reshape(-1,1)
    gamma=(params[5]+torch.rand(N)*noise_scale[5]-0.5*noise_scale[5]).reshape(-1,1)
    theta=(params[6]+torch.rand(N)*noise_scale[6]-0.5*noise_scale[6]).reshape(-1,1)
    _lambda=(params[7]+torch.rand(N)*noise_scale[7]-0.5*noise_scale[7]).reshape(-1,1)

    u_list = torch.maximum(alpha_u+beta_u+norm_scale*torch.randn((N,T+1)),torch.zeros((N,T+1)))
    dist=torch.distributions.StudentT(df=d.reshape(-1))
    t_list=dist.sample((T+1,))
    eps_list=(t_list.T)*torch.sqrt(u_list)

    
    r_list=alpha_r+beta_r*u_list+eps_list
    r_past_list=r_list[:,:T]
    r_list=r_list[:,1:]
    eps_past_list=eps_list[:,:T]

    alpha_r_list=alpha_r.expand(N,T)
    beta_r_list=beta_r.expand(N,T)
    d_list=d.expand(N,T)
    alpha_u_list=alpha_u.expand(N,T)
    beta_u_list=beta_u.expand(N,T)
    gamma_list=gamma.expand(N,T)
    theta_list=theta.expand(N,T)
    _lambda_list=_lambda.expand(N,T)




    return r_list.reshape(-1),eps_past_list.reshape(-1),r_past_list.reshape(-1),\
alpha_r_list.reshape(-1), beta_r_list.reshape(-1), d_list.reshape(-1), alpha_u_list.reshape(-1),\
        beta_u_list.reshape(-1),gamma_list.reshape(-1), theta_list.reshape(-1), _lambda_list.reshape(-1)

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


        # alpha_r = (torch.rand(1)*4*scale-2*scale).item()   #0*torch.ones(N, ) #torch.rand(N,)*scale#
        # beta_r = (torch.rand(1)*scale*1).item()#0.5*torch.ones(N, ) #torch.rand(N,)*scale
        # d = 6
        # alpha_u =  (torch.rand(1)*4*scale).item()+1#0.5*torch.ones(N, ) #torch.rand(N,)*2*scale
        # beta_u = (torch.rand(1)*scale*0.4).item()#0.2*torch.ones(N, )   #torch.rand(N,)*scale
        # gamma = (torch.rand(1)*0.4*scale).item()#0*torch.ones(N, )   #torch.rand(N,)*scale
        # theta = (torch.rand(1)*scale*0.4).item()#0*torch.ones(N, )   #torch.rand(N,)*scale
        # _lambda = (torch.rand(1)*scale).item()#4*torch.ones(N, ) #torch.rand(N,)*scale

            #alpha_r,beta_r,d,alpha_u,beta_u,gamma,theta,_lambda=0.2+np.random.randn()*0.05, 0.2+np.random.randn()*0.05, 6.0+np.random.randn(), 1+np.random.randn()*0.2, 0.4+np.random.randn()*0.1, 0.1+np.random.randn()*0.02, 0.02+np.random.randn()*0.002, 2.5
        #alpha_r,beta_r,d,alpha_u,beta_u,gamma,theta,_lambda=np.random.uniform()*10-5, 0.01+np.random.uniform()*0.5, 6.0, 0.8+np.random.uniform(), 0.01+np.random.uniform()*0.4,0.1, 0.02, 2.5
        alpha_r,beta_r,d,alpha_u,beta_u,gamma,theta,_lambda=0.1263948140321161, .032712088763023284, 6.0, 0.8715676420694289, 0.2913684368439, 0.030264586057052845, 0.008613371294834722, 2.5
        if np.random.uniform()<0.5:
            alpha_r=-np.random.uniform()*10
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

        while np.isnan(eps_templist).sum()+np.isnan(r_templist).sum()>0:
            print(f"step {i} resample")
            # alpha_r = (torch.rand(1)*10*scale-5*scale).item()   #0*torch.ones(N, ) #torch.rand(N,)*scale#
            # beta_r = (torch.rand(1)*scale).item()#0.5*torch.ones(N, ) #torch.rand(N,)*scale
            # d = 6
            # alpha_u =  (torch.rand(1)*10*scale).item()+2#0.5*torch.ones(N, ) #torch.rand(N,)*2*scale
            # beta_u = (torch.rand(1)*scale*0.2).item()#0.2*torch.ones(N, )   #torch.rand(N,)*scale
            # gamma = (torch.rand(1)*2*scale-scale).item()#0*torch.ones(N, )   #torch.rand(N,)*scale
            # theta = (torch.rand(1)*2*scale-scale).item()#0*torch.ones(N, )   #torch.rand(N,)*scale
            # _lambda = (torch.rand(1)*scale).item()#4*torch.ones(N, ) #torch.rand(N,)*scale

            #alpha_r,beta_r,d,alpha_u,beta_u,gamma,theta,_lambda=np.random.uniform()*10-5, 0.01+np.random.uniform()*0.5, 6.0, 0.8+np.random.uniform(), 0.01+np.random.uniform()*0.4,0.1, 0.02, 2.5
            alpha_r,beta_r,d,alpha_u,beta_u,gamma,theta,_lambda=0.1263948140321161, .032712088763023284, 6.0, 0.8715676420694289, 0.2913684368439, 0.030264586057052845, 0.008613371294834722, 2.5
            eps=1
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

def fine_tune(model,N=100,T=100,N_test=1000,params=(0, 0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5),noise_scale=NOISE_SCALE,norm_scale=1,\
              num_epochs = 20,batch_size = 256,loss_tolerance=10000,lr=1*1e-2,decay_step_size=5, decay_rate=0.5, device="cpu",verbose=False):
    model.train()
    
    ###Training set###
    r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda=gen_finetune_data(N=N,T=T,params=params,noise_scale=noise_scale,norm_scale=norm_scale)
    Trainset = param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda).to(device)
    dataset = TensorDataset(Trainset,r.to(device),epsilon_past.to(device),r_past.to(device),alpha_r.to(device), beta_r.to(device), d.to(device), alpha_u.to(device), beta_u.to(device),gamma.to(device), theta.to(device), _lambda.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ###Test set###
    r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda=gen_finetune_data(N=N_test,T=1,params=params)
    Testset = param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda).to(device)
    test_dataset = TensorDataset(Testset,r.to(device),epsilon_past.to(device),r_past.to(device),alpha_r.to(device), beta_r.to(device), d.to(device), alpha_u.to(device), beta_u.to(device),gamma.to(device), theta.to(device), _lambda.to(device))
    test_dataloader = DataLoader(test_dataset, batch_size=N_test, shuffle=True)

    if verbose:
        str_out=''
        for param in params:
            str_out=str_out+f" {round(param,4)}"
        print(f"------------Fine Tuning at {str_out}-------------\n")


    optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step_size, gamma=decay_rate)
    loss_epoch=[]
    test_loss_epoch=[]
    with trange(num_epochs) as t:

        for epoch in t:

            loss_list=[]
            for batch in dataloader:
                batch_data = batch[0]
                optimizer.zero_grad()
                

                outputs,outputs2,outputs3,outputs4 = model(batch_data)
                outputs,outputs2,outputs3,outputs4 =outputs.reshape(-1),outputs2.reshape(-1),outputs3.reshape(-1),outputs4.reshape(-1)
                base_dist=TruncatedNormal(loc=outputs,scale=outputs2,a=1e-4,b=100)
                modifiedbase= base_dist.rsample(sample_shape=torch.ones(1).shape).reshape(-1)
                logprob=log_likelihood_update(batch[1]-batch[4]-modifiedbase,batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8],batch[9],batch[10],batch[11])
                baselogprob=base_dist.log_prob(modifiedbase)  #torch.log(outputs4*torch.exp(base_dist.log_prob(base))+(1-outputs4)*torch.exp(base_dist2.log_prob(base)))
                loss=torch.mean(baselogprob-logprob,dim=0)#torch.mean(baselogprob-torch.log(jacobian)-logprob,dim=0)


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
                modifiedbase= base_dist.rsample(sample_shape=torch.ones(1).shape).reshape(-1).to(device)
                logprob=log_likelihood_update(batch[1]-batch[4]-modifiedbase,batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8],batch[9],batch[10],batch[11])
                baselogprob=base_dist.log_prob(modifiedbase)  
                test_loss=torch.mean(baselogprob-logprob,dim=0)
                test_loss_epoch.append(test_loss.item())


            
            loss_list=torch.tensor(loss_list)
            loss_epoch.append(torch.mean(loss_list))
            t.set_description(f"Fine Tuning Epoch: {epoch}  Train Loss:{round(torch.mean(loss_list).item(),4)} Total batch:{loss_list.shape} Test Loss:{round(test_loss.item(),4)}")
            scheduler.step()
    
    
    if verbose:
        print(f"------------Fine Tuning Complete-------------\n")
    return model


if __name__=="__main__":

    scale=1
    N=16
    T=256

    learning_rate = 1*1e-2
    num_epochs = 20
    batch_size = 256
    hidden_size= 64
    loss_tolerance=10000 #Gradually decay to 0.5*tolerance


    model = VIScaler(hidden_size=hidden_size).to(device)
    model = torch.load("./tmppth/VIScaler_test1_399_newbest.pth")
    model.train()


    params1=(0.2, 0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5)
    params2=(0.1263948140321161, .032712088763023284, 6.0, 0.8715676420694289, 0.2913684368439, 0.030264586057052845, 0.008613371294834722, 2.5)


    r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda=gen_finetune_data(N=N,T=T,params=params1)
    print("-----------Train Dataset------------")
    print("r",r.max())
    Trainset = param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda).to(device)
    #print(Trainset)
    dataset = TensorDataset(Trainset,r.to(device),epsilon_past.to(device),r_past.to(device),alpha_r.to(device), beta_r.to(device), d.to(device), alpha_u.to(device), beta_u.to(device),gamma.to(device), theta.to(device), _lambda.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(Trainset[:,0].min(),Trainset[:,0].max())

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

    Testset = param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda).to(device)
    #print(Trainset)
    test_dataset = TensorDataset(Testset,r.to(device),epsilon_past.to(device),r_past.to(device),alpha_r.to(device), beta_r.to(device), d.to(device), alpha_u.to(device), beta_u.to(device),gamma.to(device), theta.to(device), _lambda.to(device))
    test_dataloader = DataLoader(test_dataset, batch_size=N, shuffle=True)



    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001,)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)
    loss_epoch=[]
    test_loss_epoch=[]

    with trange(num_epochs) as t:

        for epoch in t:

            loss_list=[]
            for batch in dataloader:
                batch_data = batch[0]
                optimizer.zero_grad()
                

                outputs,outputs2,outputs3,outputs4 = model(batch_data)
                outputs,outputs2,outputs3,outputs4 =outputs.reshape(-1),outputs2.reshape(-1),outputs3.reshape(-1),outputs4.reshape(-1)
                base_dist=TruncatedNormal(loc=outputs,scale=outputs2,a=1e-4,b=100)
                modifiedbase= base_dist.rsample(sample_shape=torch.ones(1).shape).reshape(-1)
                logprob=log_likelihood_update(batch[1]-batch[4]-modifiedbase,batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8],batch[9],batch[10],batch[11])
                baselogprob=base_dist.log_prob(modifiedbase)  #torch.log(outputs4*torch.exp(base_dist.log_prob(base))+(1-outputs4)*torch.exp(base_dist2.log_prob(base)))
                loss=torch.mean(baselogprob-logprob,dim=0)#torch.mean(baselogprob-torch.log(jacobian)-logprob,dim=0)


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
                modifiedbase= base_dist.rsample(sample_shape=torch.ones(1).shape).reshape(-1).to(device)

                logprob=log_likelihood_update(batch[1]-batch[4]-modifiedbase,batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8],batch[9],batch[10],batch[11])

                
                baselogprob=base_dist.log_prob(modifiedbase)  #torch.log(outputs4*torch.exp(base_dist.log_prob(base))+(1-outputs4)*torch.exp(base_dist2.log_prob(base)))

                test_loss=torch.mean(baselogprob-logprob,dim=0)#torch.mean(baselogprob-torch.log(jacobian)-logprob,dim=0)
                test_loss_epoch.append(test_loss.item())


            
            loss_list=torch.tensor(loss_list)
            #print(f"Loss:{torch.mean(loss_list)} Total batch:{loss_list.shape} Loss:{loss.item()}")
            loss_epoch.append(torch.mean(loss_list))
            t.set_description(f"Epoch: {epoch}  Train Loss:{torch.mean(loss_list)} Total batch:{loss_list.shape} Test Loss:{test_loss}")
            if epoch%10==9:
                plt.figure()
                plt.yscale("log")
                plt.xscale("log")
                plt.plot(test_loss_epoch)
                plt.savefig(f"./figs/Loss_test1_Epoch1-{epoch}.png")
                torch.save(model,f"./tmppth/VIScaler_test1_{epoch}.pth")
            scheduler.step()


