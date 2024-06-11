import numpy as np
from scipy.stats import t, expon, norm
import matplotlib.pyplot as plt
import torch.distributions as tdist
import pdb
import torch
from torch.distributions import Normal
import torch.nn as nn

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
        self.fc21 = nn.Linear(hidden_size, 1)
        self.fc22 = nn.Linear(hidden_size, 1)
        self.fc23 = nn.Linear(hidden_size, 1)
        self.fc24 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x1 = self.fc21(x)
        x2 = self.fc22(x)
        x3 = self.fc23(x)
        x4 = self.fc24(x)
        x1 = 1e-2+torch.sigmoid(x1)
        x2 = 1e-4+0.4*torch.sigmoid(x2)
        x3 = 1e-4+0.4*torch.sigmoid(x3)
        x4 = 1e-4+torch.sigmoid(x4)
        return x1,x2,x3,x4

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

class TEST_SAMPLER:
    """test sampler"""
    ESS_list = []
    sample_num = 1
    def __init__(self, T, params,path="VIScaler_test1_99.pth"):
        self.alpha_r, self.beta_r, self.d, self.alpha_u, self.beta_u,self.gamma, self.theta, self._lambda = params
        # print(params)
        self.params = params
        self.T = T
                
        self.model=VIScaler(hidden_size=16)
        self.model=torch.load(path)
        self.model.eval()
        self.base_dist=tdist.Gamma(1.5,0.75)
        self.base_dist2=HalfNormal(1)

    def log_likelihood_update(self,epsilon,r,epsilon_past,r_past):
        ''' Compute log joint likelihood of l=log p(eps_t,r_t|eps_{t-1},r_{t-1})'''
        #Input:  epsilon  (n,) epsilon_past  (n,)  r (1,) r_past (1,)
        #Output: log_prob (n,)

        prior=0 #do we need prior on r_0, eps_0?

        ''' Calculate u, w, nu and eta'''
        #u=(r-epsilon-self.alpha_r)/self.beta_r
        u_past= (r_past-epsilon_past-self.alpha_r)/self.beta_r

        w=self.alpha_u+self.beta_u*u_past+(self.gamma+self.theta*(epsilon_past<0))*(epsilon_past**2)

        nu=np.sqrt(self.beta_r/(r-epsilon-self.alpha_r+1e-6))*epsilon
        nu=np.where(np.isnan(nu),1e10,nu)
        eta=(r-epsilon-self.alpha_r)/self.beta_r-w
        #eta=np.maximum(eta,1e-7)
        #print(f"eps:{epsilon[:10]}\n eps_past:{epsilon_past[:10]}\n r:{r} r_past:{r_past} etamin:{eta.min()}\n w:{w[:10]}, nu:{nu[:10]}\n eta:{eta[:10]}\n")
        #print(eta.min())
        eta=np.where(np.isnan(eta),1e6,eta)
        #eta=np.where(eta<0,1e8,eta)
        #eta=(eta>=0)*eta+(eta<=0)*1e-6
        #assert eta.min()>=0 #eta should follow exponential distribution

        #logp_exp=expon.logpdf(eta, scale=1/self._lambda)
        logp_exp=norm.logpdf(eta, scale=0.5)
        logp_t=t.logpdf(nu,self.d)


        log_joint=logp_exp+logp_t -0.5*(np.log(self.beta_r)+np.log(r-epsilon-self.alpha_r))+prior
        #log_joint=np.where(np.isnan(log_joint),-1e10,log_joint)
        #print(log_joint)
        return log_joint
        
    def sample(self, sample_num:int, r, exp_scale=1, resample_thre=0.1):
        self.ESS_list = []
        self.sample_num = sample_num
        
        samples = np.zeros((sample_num,self.T))
        log_weights = np.ones(sample_num)
        eps=np.zeros(sample_num)
        r_past=self.alpha_r
        for i in range(self.T): 
            # print(f"----------STEP{i}----------")
            #pdb.set_trace()
            rr=r[i]
            eps_past=eps.copy()
            u_past= (r_past-eps_past-self.alpha_r)/self.beta_r
            w=self.alpha_u+self.beta_u*u_past+(self.gamma+self.theta*(eps_past<0))*(eps_past**2)
            eps,log_density = self.policy(eps_past, rr, w, exp_scale,r_past)
            # print(f"step{i}\n eps:{eps}\n eps_past:{eps_past} \n rr:{rr} w:{w} exp_scale:{exp_scale}\n")
            log_weights += self.log_likelihood_update(eps,rr,eps_past,r_past)-log_density          #self.log_policy_density(eps, rr, w, exp_scale,r_past)

            
            r_past=rr
            samples[:,i]=eps
            weights=np.exp(log_weights)
            weights=weights/weights.sum()
            
            ESS = 1/np.sum(np.power(weights, 2))
            self.ESS_list.append(ESS)
            if ESS < sample_num*resample_thre:
                samples[:,:i] = self.resample(samples[:,:i], weights)
                weights = np.ones(sample_num)/sample_num
                log_weights=np.zeros(sample_num)
                eps=samples[:,i]
        return samples, weights
    
    def plot_ESS(self, y_high=0, title=""):
        if y_high == 0:
            y_high = self.sample_num
        plt.plot(range(self.T), self.ESS_list)
        plt.ylim(0, y_high)
        plt.xlim(0, self.T)
        plt.ylabel("ESS")
        if title != "":
            plt.title(title)
        plt.show()
        plt.clf()
    
    def resample(self, samples, weights):
        index = np.random.choice(list(range(len(weights))), p=weights, size=(len(weights)))
        return samples[index]
    
    def policy(self, eps_past, rr, w, exp_scale,r_past):
        # print("exponentials:",expon.rvs(scale=exp_scale,size=self.sample_num))
        # print("values:",rr,self.alpha_r,self.beta_r*w,self.beta_r*np.random.exponential(scale=exp_scale,size=self.sample_num))
        #print("generate eps:",rr-self.alpha_r-self.beta_r*w-self.beta_r*np.random.exponential(scale=exp_scale,size=self.sample_num))
        inputs=param_to_input(torch.tensor(rr),torch.tensor(eps_past),torch.tensor(r_past),torch.tensor(self.alpha_r),torch.tensor(self.beta_r),
                              torch.tensor(self.d),torch.tensor(self.alpha_u),torch.tensor(self.beta_u),torch.tensor(self.gamma),
                              torch.tensor(self.theta),torch.tensor(self._lambda))
        outputs,outputs2,outputs3,outputs4 = self.model(inputs)
        outputs,outputs2,outputs3,outputs4 =outputs.reshape(-1),outputs2.reshape(-1),outputs3.reshape(-1),outputs4.reshape(-1)
        assert torch.tensor(eps_past).shape[0]==outputs.shape[0]
        #outputs=torch.tensor(0.4).expand(inputs.shape[0],)
        base1=self.base_dist.sample((inputs.shape[0],)).reshape(-1)
        base2=self.base_dist2.sample((inputs.shape[0],)).reshape(-1)
        random_tensor = torch.rand_like(outputs4)

        base = torch.where(random_tensor < outputs4, base1, base2)
        baselogprob=torch.log(outputs4*torch.exp(self.base_dist.log_prob(base))+(1-outputs4)*torch.exp(self.base_dist2.log_prob(base)))
        print(outputs2)
        modifiedbase=outputs*base+outputs2*base**1.5+outputs3*base**0.5
        jacobian=outputs+1.5*base**0.5*outputs2+0.5*outputs3*base**(-0.5)
        sample=torch.tensor(rr)-torch.tensor(self.alpha_r)-modifiedbase
        return sample.detach().numpy(),(baselogprob-torch.log(jacobian)).detach().numpy()
        #return rr-self.alpha_r-expon.rvs(scale=exp_scale,size=self.sample_num)
    #np.random.exponential(scale=exp_scale,size=self.sample_num) #a simple policy
    
    def policy_old(self, eps_past, rr, w, exp_scale,r_past):
        eps=rr-self.alpha_r-expon.rvs(scale=exp_scale,size=self.sample_num)
        return eps, expon.logpdf((rr-self.alpha_r-eps),scale=exp_scale)
    
    def log_policy_density(self, eps, rr, w, exp_scale,r_past):
        #Now combined in policy()


        # print("exponential values:",(rr-self.alpha_r-self.beta_r*w-eps)/self.beta_r)
        # print("logpdf:",expon.logpdf((rr-self.alpha_r-self.beta_r*w-eps)/self.beta_r,scale=exp_scale))
        return expon.logpdf((rr-self.alpha_r-eps),scale=exp_scale)





if __name__=="__main__":
    r=np.load("./r.npy")
    print(r.shape)
    params=(0.2, 0.2, 6.0, 1.0, 0.4, 0.1, 0.02, 2.5)
    sampler=TEST_SAMPLER(20,params,path='VIScaler_test1_174_loss_5790.3994140625.pth')
    samples,weights=sampler.sample(10000,r,exp_scale=0.4)
    sampler.plot_ESS()
    print(r.shape,samples.shape,weights.shape)
    print(samples)
    i=10
    index = np.random.choice(list(range(len(weights))), p=weights, size=(len(weights)))
    plt.hist((samples[:,i])[index], density=True, bins=40, label="sampled")
    plt.legend()
    plt.show()
    #print(weights)
