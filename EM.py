import torch
# import line_profiler
# profile = line_profiler.LineProfiler()
import numpy as np
from torch.distributions import StudentT, Exponential, Normal
import sys
#sys.path.append( '/Users/hengyuf/Library/CloudStorage/OneDrive-北京大学/大学学习/数学/Bayes/VSMCG' )
from sampler_newmodel import *
#from sampler_gaussian import *
from tqdm import tqdm, trange
import time
import os
import torch.optim as optim




def generate_grid_samples(n, T,lower=0,upper=0):
    """
    Generates grid samples of size n that cover most of the [-1, 1]^T space.

    Parameters:
    n (int): Number of samples to generate.
    T (int): Number of dimensions.

    Returns:
    torch.Tensor: Tensor of shape (n, T) containing the generated samples.
    """
    # Determine the number of points per dimension (rounded to the nearest integer)
    points_per_dim = int(torch.ceil(torch.tensor(n ** (1 / T))).item())

    # Create a grid of points in each dimension
    grid_list = [torch.linspace(lower[t],upper[t], points_per_dim)for t in range(T)] 

    # Generate all combinations of grid points (this can be very large)
    mesh = torch.meshgrid(grid_list, indexing='ij')
    grid_samples = torch.stack(mesh, dim=-1).reshape(-1, T)

    # Shuffle the grid samples to introduce some randomness
    perm = torch.randperm(grid_samples.size(0))
    grid_samples = grid_samples[perm]

    # If there are more grid samples than required, select n of them
    if grid_samples.size(0) > n:
        grid_samples = grid_samples[:n]
    
    # If there are fewer grid samples than required, repeat some of them to reach n
    while grid_samples.size(0) < n:
        additional_samples = grid_samples[:min(grid_samples.size(0), n - grid_samples.size(0))]
        grid_samples = torch.cat([grid_samples, additional_samples], dim=0)

    return grid_samples


#print('\n'*10)

class EM:
    def __init__(self, T=100, _alpha_r=0, _beta_r=1, _d=1, _alpha_u=0, _beta_u=1, _gamma=1, _theta=0, __lambda=1,rfilename="./r.npy"):
        self.alpha_r = torch.tensor(_alpha_r, dtype=torch.float64, requires_grad=True)
        self.beta_r = torch.tensor(_beta_r, dtype=torch.float64, requires_grad=True)
        self.alpha_u = torch.tensor(_alpha_u, dtype=torch.float64, requires_grad=True)
        self.beta_u = torch.tensor(_beta_u, dtype=torch.float64, requires_grad=True)
        self.gamma = torch.tensor(_gamma, dtype=torch.float64, requires_grad=True)
        self.theta = torch.tensor(_theta, dtype=torch.float64, requires_grad=True)
        self._lambda = torch.tensor(__lambda, dtype=torch.float64, requires_grad=True)
        self.d = torch.tensor(_d, dtype=torch.float64, requires_grad=True)
        #self.parameters=[self.alpha_r, self.beta_r, self.alpha_u, self.beta_u, self.gamma, self.theta, self._lambda, self.d]
        #self.names=["alpha_r", "beta_r", "alpha_u", "beta_u", "gamma", "theta", "_lambda", "d"]
        self.parameters=[ self.theta]
        self.names=["theta"]
        self.T=T
        self.r=np.load(rfilename)
        self.r=torch.tensor(self.r).reshape(1,-1)
        assert self.T ==self.r.shape[1]
    def singularlikelihood(self,epsilon):
        ''' Compute log joint likelihood of l=log p(eps_1,...,eps_T,r_1,...r_T)'''
        #Input:  epsilon  (n,T)
        #Output: log_prob (n,)

        n,T=epsilon.shape
        prior=0 #do we need prior on r_0, eps_0?

        ''' Calculate u, w, nu and eta'''
        epsilon_shift=torch.zeros(n,T)
        epsilon_shift[:,1:]=epsilon[:,:T-1] #eps_{t-1}
        u=(self.r-epsilon-self.alpha_r)/self.beta_r
        u_shift=torch.zeros_like(u) #u_{t-1}
        u_shift[:,1:]=u[:,:T-1]
        w=self.alpha_u+self.beta_u*u_shift+(self.gamma+self.theta*(epsilon_shift<0))*(epsilon_shift**2)

        nu=torch.sqrt(self.beta_r/(self.r-epsilon-self.alpha_r+1e-4))*epsilon
        eta=(self.r-epsilon-self.alpha_r)/self.beta_r-w

        #print(f"Test alpha:{self.alpha_r} nu:{torch.mean(nu)} eta: {torch.mean(eta)}")

        if torch.isnan(nu).int().sum()+torch.isnan(eta).int().sum():
            print("NaN encountered, total NaNs:",torch.isnan(nu).int().sum()+torch.isnan(eta).int().sum())
            print("alpha_r:",self.alpha_r)
            raise NotImplementedError
            
        #nu=torch.where(torch.isnan(nu),1e8,nu)

        #eta=torch.where(torch.isnan(eta),1e5,eta)
        #eta=torch.where(eta<0,1e4,eta)
        #eta=torch.maximum(torch.zeros_like(eta),eta)+1e-8
        #print(eta)
        # if eta.min()<0:
        #     print("eta<0",(eta<0).int().sum(dim=0))
        #assert eta.min()>=0 #eta should follow exponential distribution

        ''' Calculate each component of the log-likelihood'''
        t_distr=StudentT(self.d)
        exp_distr=Normal(loc=0,scale=0.5) #remains to be checked: lambda or 1/lambda
        
        #nu=torch.where(torch.isnan(nu),1e5,nu)
        #nu=(torch.abs(nu)<1e10)*nu+(torch.abs(nu)>=1e10)*1e10
        logp_t=t_distr.log_prob(nu)


        logp_exp=exp_distr.log_prob(eta)
        
        log_joint=torch.sum(logp_t+logp_exp -0.5*(torch.log(self.beta_r)+torch.log(self.r-epsilon-self.alpha_r)), dim=-1)+prior
        if torch.isnan(log_joint).int().sum():
            print("joint probability meets -INF, counts",torch.isnan(log_joint).int().sum())
            self.state_EM()
            raise NotImplementedError
            

        

        return log_joint
    
    def compute_truth_likelihood(self,n=100000,radius=10,lower=0,upper=0):
        ''' Compute log joint likelihood of l=log p(eps_1,...,eps_T,r_1,...r_T)'''
        #Input:  epsilon  (n,T)
        #Output: log_prob (n,)
        epsilon=generate_grid_samples(n, self.T,lower=lower,upper=upper)

        T=self.T
        prior=0 #do we need prior on r_0, eps_0?

        ''' Calculate u, w, nu and eta'''
        epsilon_shift=torch.zeros(n,T)
        epsilon_shift[:,1:]=epsilon[:,:T-1] #eps_{t-1}
        u=(self.r-epsilon-self.alpha_r)/self.beta_r
        u_shift=torch.zeros_like(u) #u_{t-1}
        u_shift[:,1:]=u[:,:T-1]
        w=self.alpha_u+self.beta_u*u_shift+(self.gamma+self.theta*(epsilon_shift<0))*(epsilon_shift**2)


        nu=torch.where(self.r-epsilon-self.alpha_r>=0,torch.sqrt(self.beta_r/(self.r-epsilon-self.alpha_r+1e-6))*epsilon,1e8)  
        eta=(self.r-epsilon-self.alpha_r)/self.beta_r-w


        ''' Calculate each component of the log-likelihood'''
        t_distr=StudentT(self.d)
        exp_distr=Normal(loc=0,scale=0.5) #remains to be checked: lambda or 1/lambda
        logp_t=t_distr.log_prob(nu)
        #nu=torch.where(torch.isnan(nu),1e5,nu)
        #nu=(torch.abs(nu)<1e10)*nu+(torch.abs(nu)>=1e10)*1e10
        logp_exp=exp_distr.log_prob(eta)
        #print("logp_t:",logp_t)
        #print("logp_exp",logp_exp)
        logp_numer=torch.where(self.r-epsilon-self.alpha_r>=0,torch.log(self.beta_r)+torch.log(self.r-epsilon-self.alpha_r),1e10)


        log_joint=torch.sum(logp_t+logp_exp -0.5*logp_numer, dim=-1)+prior

        radius=upper-lower
        #print("radius:",radius)
        #print("log_joint:",log_joint)
        return torch.log(torch.sum(torch.exp(log_joint))) +(torch.sum(torch.log(radius)))-np.log(n)
    
    def call_sampler(self,n,verbose=False):
        #remember to add .item() at final version
        params=(self.alpha_r.item(), self.beta_r.item(), self.d.item(), self.alpha_u.item(), self.beta_u.item(), self.gamma.item(), self.theta.item(), self._lambda.item())
        # sampler = TEST_SAMPLER(self.T, params)
        sampler=TEST_SAMPLER(self.T, params,path='./pth/VIScaler_test1_199.pth')
        if verbose:
            print("Call sampler with params",params)
        samples, weights=sampler.sample(n, np.array(self.r.T),resample_thre=0.2)
        weights,samples=torch.tensor(weights),torch.tensor(samples)
        '''Check sample quality'''
        if torch.max(torch.abs(samples))>1e5:
            print("overflow encountered:checking samples and weights:")
            print("samples:",samples)
            print("weights:",weights)
            self.state_EM()
        if torch.isnan(weights).int().sum()+torch.isnan(samples).int().sum():
            print("NaN encountered:checking samples and weights:")
            print("samples:",samples)
            print("weights:",weights)
            self.state_EM()



        #print("Weights&Samples",weights,samples)
        return weights,samples
    def upd_param(self,lr=1e-4,verbose=True):
        with torch.no_grad():
            for i,param in enumerate(self.parameters):
                param += lr * param.grad
                param.grad.zero_()
        #print(self.parameters)
    def log_total(self,weights,epsilon):
        likelihood=self.singularlikelihood(epsilon)
        #print("Sing l",likelihood)
        normed=likelihood-torch.min(likelihood).detach()+np.log(weights) #item for removing normalization from ???gradients

        return torch.log(torch.sum(torch.exp(normed)))
    def state_EM(self):
        for i,param in enumerate(self.parameters):
            print(self.names[i],param.item())

    
    def optimize(self,num_steps=10,n=100000,batch_size=128,max_epoch=2,init_lr=1e-2,decay=0.9,compute_truth_l=True):
        lr=init_lr
        self.Likelihood_list=[]

        with trange(num_steps) as t:
            for _ in t:
                optimizer =torch.optim.Adam(self.parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
                scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.5)
                
                weights,epsilon=self.call_sampler(n)
                
                batch_total_num=n//batch_size
                #print(_)
                update_count=0

                for epoch in range(max_epoch):
                    ll_list=[]
                    for batch_num in range(batch_total_num):


                        eps_batch=epsilon[batch_num*batch_size:(batch_num+1)*batch_size]
                        weights_batch=weights[batch_num*batch_size:(batch_num+1)*batch_size]

                        if torch.min(self.r-eps_batch-self.alpha_r)>-1e-5:
                            optimizer.zero_grad()
                            likelihood= self.log_total(weights_batch,eps_batch)
                            
                            
                            
                            loss=-likelihood#+50*(self.alpha_r-0.1)**2 #
                            truth_l=likelihood
                            if compute_truth_l: #Only works with small T?
                                truth_l=self.compute_truth_likelihood(n=1000000,lower=torch.ones(self.T)*(-5),upper=(self.r-self.alpha_r).reshape(self.T))
                            ll_list.append(truth_l.detach().numpy())
                            loss.backward()
                            #self.upd_param(lr=2*1e-5,verbose=verbose)
                            optimizer.step()
                            
                            update_count+=1
                            
                            with torch.no_grad():
                                if self.alpha_r<-2:
                                    self.alpha_r=self.alpha_r-self.alpha_r-2
                                if self.beta_r<0:
                                    self.beta_r-=self.beta_r+1e-2
                                # self.alpha_r=torch.maximum(torch.zeros_like(self.alpha_r),self.alpha_r)
                                # self.beta_r=torch.maximum(1e-2*torch.ones_like(self.beta_r),self.beta_r)
                            #time.sleep(0.01)
                            t.set_description(f"EM Step: {_} Epoch: {epoch} update count:{update_count} lr:{"{:.2E}".format(lr)} L:{"{:.4E}".format(truth_l)} beta_r:{round(self.beta_r.item(),4)} alpha_r:{round(self.alpha_r.item(),4)} alpha_u: {round(self.alpha_u.item(),4)} beta_u: {round(self.beta_u.item(),4)} theta:{round(self.theta.item(),4)}")
                        else:
                            #time.sleep(0.01)
                            t.set_description(f"EM Step: {_} Epoch: {epoch} min:{torch.min(self.r-eps_batch-self.alpha_r)} lr:{"{:.2E}".format(lr)} beta_r:{round(self.beta_r.item(),4)} alpha_r:{round(self.alpha_r.item(),4)} alpha_u: {round(self.alpha_u.item(),4)} beta_u: {round(self.beta_u.item(),4)} theta:{round(self.theta.item(),4)}")



                        if torch.isnan(self.alpha_r).int():
                            print("NaN parameters after GD")
                            self.state_EM()
                            raise NotImplementedError
                    if epoch==max_epoch-1:
                        self.Likelihood_list.append(np.mean(np.array(ll_list)))
                    #time.sleep(0.1)
                    scheduler.step()
                if  _>=1:
                    torch.save(self,f"./pth/EM_model_step{_}_loss_{self.Likelihood_list[-1]}.pt")
                lr*=decay

                # if _ >=1:
                #     t.set_description(f"EM Step: {_} Epoch: {epoch}  Likelihood:{self.compute_truth_likelihood(n=1024**2,lower=torch.ones(self.T)*(-10),upper=(self.r-self.alpha_r).reshape(self.T))}")
                #     time.sleep(1)
                #     #self.state_EM()
            
                
    
        

    

if __name__=="__main__":
    #EM_sampler=EM(10,0.2, 0.2, 6.0, 0.6, 0.4, 0.1, 0.02, 2.5,rfilename="./r.npy")
    T=300
    r=np.load("./data/r.npy")
    EM_sampler=EM(T, 0.2, 0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5,rfilename="./data/r.npy")

    # alpha_r_list=torch.linspace(0.0,5.0,20)
    
    # for alpha_r in tqdm(alpha_r_list):
        
    #     EM_sampler.alpha_r=alpha_r
    #     print("alpha_r:",EM_sampler.alpha_r)
    #     weights,epsilon=EM_sampler.call_sampler(100000)
    #     print("mean of eps:",torch.mean(epsilon,dim=0))
    #     ll_list=[]
    #     aa_list=[]
    #     for test_alpha_r in alpha_r_list:
    #         if test_alpha_r<alpha_r:
    #             aa_list.append(test_alpha_r)
    #             EM_sampler.alpha_r=test_alpha_r
    #             ll_list.append(torch.sum(EM_sampler.singularlikelihood(epsilon)+weights).detach())
    #     if len(ll_list)>0:
    #         plt.figure()
    #         plt.xlabel(r"$\alpha_r$")
    #         plt.ylabel("Likelihood")
    #         plt.title(f"post likelihood with alpha_r={alpha_r}.png")
    #         plt.plot(aa_list,ll_list)
    #         plt.savefig(f"./figs/post likelihood with alpha_r={alpha_r}.png")





    EM_sampler.optimize(num_steps=500,n=2048,batch_size=2048,max_epoch=1,init_lr=3e-3,decay=0.996,compute_truth_l=False)
    plt.figure()
    plt.plot(EM_sampler.Likelihood_list)
    plt.xlabel("EM steps")
    plt.ylabel("Log likelihood")
    plt.xscale("log")
    plt.show()
    torch.save(EM_sampler,"./EM_model.pt")
    #print(EM_sampler.compute_truth_likelihood(n=1000000,lower=-5*np.ones(2),upper=5*np.ones(2)))
    # profile.print_stats()
