import torch
import numpy as np
from torch.distributions import StudentT, Exponential
from sampler import TEST_SAMPLER

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
        self.parameters=[self.alpha_r, self.beta_r, self.alpha_u, self.beta_u, self.gamma, self.theta, self._lambda, self.d]
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
        nu=torch.sqrt(self.beta_r/(self.r-epsilon-self.alpha_r))*epsilon
        eta=(self.r-epsilon-self.alpha_r)/self.beta_r-w
        print(eta,epsilon)
        assert eta.min()>0 #eta should follow exponential distribution

        ''' Calculate each component of the log-likelihood'''
        t_distr=StudentT(self.d)
        exp_distr=Exponential(self._lambda) #remains to be checked: lambda or 1/lambda
        logp_t=t_distr.log_prob(nu)
        logp_exp=exp_distr.log_prob(eta)
        log_joint=torch.sum(logp_t+logp_exp -0.5*(torch.log(self.beta_r)+torch.log(self.r-epsilon-self.alpha_r)), dim=-1)+prior


        return log_joint
    def call_sampler(self,n):
        #remember to add .item() at final version
        params=(self.alpha_r.item(), self.beta_r.item(), self.d.item(), self.alpha_u.item(), self.beta_u.item(), self.gamma.item(), self.theta.item(), self._lambda.item())
        sampler = TEST_SAMPLER(self.T, params)
        samples, weights=sampler.sample(n, np.array(self.r.T))
        return torch.tensor(weights),torch.tensor(samples)
    def upd_param(self,lr=0.01):
        with torch.no_grad():
            for param in self.parameters:
                param -= lr * param.grad #check + or -
                param.grad.zero_()
    def log_total(self):
        n=10
        weights,epsilon=self.call_sampler(n)
        likelihood=self.singularlikelihood(epsilon)
        normed=likelihood-torch.sum(likelihood).item()/n+torch.log(weights) #item for removing normalization from gradients
        return torch.log(torch.sum(torch.exp(normed)))
    def optimize(self,num_steps=10):
        for _ in range(num_steps):
            likelihood=self.log_total()
            likelihood.backward()
            self.upd_param(lr=0.01)
    
        

    
    

EM_sampler=EM(5,0.2, 0.2, 6.0, 0.6, 0.4, 0.1, 0.02, 2.5,rfilename="./Bayes_temp/r.npy")
EM_sampler.optimize()
print(EM_sampler.parameters)
