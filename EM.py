import torch
import numpy as np
from torch.distributions import StudentT, Exponential, Normal
from sampler import *
from tqdm import tqdm

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
        # self.parameters=[self.alpha_r, self.beta_r, self.alpha_u, self.beta_u, self.gamma, self.theta, self._lambda, self.d]
        # self.names=["alpha_r", "beta_r", "alpha_u", "beta_u", "gamma", "theta", "_lambda", "d"]
        self.parameters=[self.alpha_u]
        self.names=["alpha_u"]
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

        nu=torch.sqrt(self.beta_r/(self.r-epsilon-self.alpha_r+1e-6))*epsilon
        eta=(self.r-epsilon-self.alpha_r)/self.beta_r-w

        if torch.isnan(nu).int().sum()+torch.isnan(eta).int().sum():
            #print("NaN encountered, total NaNs:",torch.isnan(nu).int().sum()+torch.isnan(eta).int().sum())
            #self.state_EM()
            nu=torch.where(torch.isnan(nu),0,nu)
            eta=torch.where(torch.isnan(eta),0,eta)
            
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
            #print("joint probability meets NaNs, counts",torch.isnan(log_joint).int().sum())
            log_joint=torch.where(torch.isnan(log_joint),-1e6,log_joint)

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
        
        #nu=torch.where(torch.isnan(nu),1e5,nu)
        #nu=(torch.abs(nu)<1e10)*nu+(torch.abs(nu)>=1e10)*1e10
        try:
            logp_t=t_distr.log_prob(nu)
        except:
            print("error encountered in computing the truth likelihood:",nu.min(),nu.max())
            self.state_EM()
            logp_t=torch.ones_like(nu)*(-100000)
        logp_exp=exp_distr.log_prob(eta)
        #print("logp_t:",logp_t)
        #print("logp_exp",logp_exp)
        logp_numer=torch.where(self.r-epsilon-self.alpha_r>=0,torch.log(self.beta_r)+torch.log(self.r-epsilon-self.alpha_r),1e10)


        log_joint=torch.sum(logp_t+logp_exp -0.5*logp_numer, dim=-1)+prior

        radius=torch.tensor(upper-lower)
        return torch.log(torch.sum(torch.exp(log_joint))/n) +(torch.sum(torch.log(radius)))

    def call_sampler(self,n,verbose=False):
        #remember to add .item() at final version
        params=(self.alpha_r.item(), self.beta_r.item(), self.d.item(), self.alpha_u.item(), self.beta_u.item(), self.gamma.item(), self.theta.item(), self._lambda.item())
        sampler = TEST_SAMPLER(self.T, params)
        if verbose:
            print("Call sampler with params",params)
        samples, weights=sampler.sample(n, np.array(self.r.T),exp_scale=1/self._lambda.item(),resample_thre=0.8)
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
        normed=likelihood+torch.log(weights) #item for removing normalization from gradients
        return torch.log(torch.mean(torch.exp(normed)))
    def state_EM(self):
        for i,param in enumerate(self.parameters):
            print(self.names[i],param.item(),param.grad.item())
    def optimize(self,num_steps=10,n=100000,batch_size=128):


        
        
        for _ in tqdm(range(num_steps)):
            optimizer =torch.optim.Adam(self.parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001, amsgrad=False)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.5)
            
            weights,epsilon=self.call_sampler(n)
            
            batch_total_num=n//batch_size
            #print(_)
            update_count=0
            
            for batch_num in range(batch_total_num):
                verbose=False
                if _ % 5==1 and batch_num %100==1:
                    print(f"Innerstep:{batch_num} (fake)Likelihood:",likelihood.item())
                    #self.state_EM()

                eps_batch=epsilon[batch_num*batch_size:(batch_num+1)*batch_size]
                weights_batch=weights[batch_num*batch_size:(batch_num+1)*batch_size]

                if torch.min(self.r-eps_batch-self.alpha_r)>-1e-5:
                    likelihood=self.log_total(weights_batch,eps_batch)
                    optimizer.zero_grad()
                    loss=-likelihood
                    loss.backward()
                    #self.upd_param(lr=2*1e-5,verbose=verbose)
                    optimizer.step()
                    scheduler.step()
                    update_count+=1
                if torch.isnan(self.alpha_r).int():
                    print("NaN parameters after GD")
                    self.state_EM()
                    return
            if _ %5==1:
                print(f"------STEP {_}---------")

                lower=np.min(np.array(epsilon),axis=0).reshape(-1)
                upper=np.max(np.array(epsilon),axis=0).reshape(-1)
                print("Truth Likelihood:",self.compute_truth_likelihood(n=1024**2,lower=lower-5,upper=upper+5))
                print("range of eps:",lower,upper)
                print(f"Update counts:{update_count} Batch_num:{batch_total_num}")
                self.state_EM()
            
                
    
        

    
    
if __name__=="__main__":
    #EM_sampler=EM(10,0.2, 0.2, 6.0, 0.6, 0.4, 0.1, 0.02, 2.5,rfilename="./r.npy")
    EM_sampler=EM(2,3, 0.2, 6.0, 2, 0.4, 0.1, 0.02, 2.5,rfilename="./r.npy")
    EM_sampler.optimize(num_steps=200,n=1024*1000,batch_size=8192)
    #print(EM_sampler.compute_truth_likelihood(n=1000000,lower=-5*np.ones(2),upper=5*np.ones(2)))
