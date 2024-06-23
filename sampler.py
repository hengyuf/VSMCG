import numpy as np
from scipy.stats import t, expon, norm
import matplotlib.pyplot as plt
import pdb
import torch
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
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x=  1e-2+torch.sigmoid(x)
        return x

class TEST_SAMPLER:
    """test sampler"""
    ESS_list = []
    sample_num = 1
    def __init__(self, T, params,path="VIScaler_best.pth"):

        self.alpha_r, self.beta_r, self.d, self.alpha_u, self.beta_u,self.gamma, self.theta, self._lambda = params
        # print(params)
        self.params = params
        self.T = T
                
        self.model=VIScaler(hidden_size=16)
        self.model=torch.load(path)
        self.model.eval()

    def log_likelihood_update(self,epsilon,r,epsilon_past,r_past):
        ''' Compute log joint likelihood of l=log p(eps_t,r_t|eps_{t-1},r_{t-1})'''
        #Input:  epsilon  (n,) epsilon_past  (n,)  r (1,) r_past (1,)
        #Output: log_prob (n,)

        prior=0 #do we need prior on r_0, eps_0?

        ''' Calculate u, w, nu and eta'''
        #u=(r-epsilon-self.alpha_r)/self.beta_r
        u_past= (r_past-epsilon_past-self.alpha_r)/self.beta_r

        w=self.alpha_u+self.beta_u*u_past+(self.gamma+self.theta*(epsilon_past<0))*(epsilon_past**2)

        #print("min r-eps-alpha_r:",np.min(r-epsilon-self.alpha_r))

        nu=np.sqrt(self.beta_r/(r-epsilon-self.alpha_r+1e-6))*epsilon
        #nu=np.where(np.isnan(nu),1e10,nu)
        eta=(r-epsilon-self.alpha_r+1e-6)/self.beta_r-w
        #eta=np.maximum(eta,1e-7)
        #print(f"eps:{epsilon[:10]}\n eps_past:{epsilon_past[:10]}\n r:{r} r_past:{r_past} etamin:{eta.min()}\n w:{w[:10]}, nu:{nu[:10]}\n eta:{eta[:10]}\n")
        #print(eta.min())
        #eta=np.where(np.isnan(eta),1e6,eta)
        #eta=np.where(eta<0,1e8,eta)
        #eta=(eta>=0)*eta+(eta<=0)*1e-6
        #assert eta.min()>=0 #eta should follow exponential distribution

        #logp_exp=expon.logpdf(eta, scale=1/self._lambda)
        logp_exp=norm.logpdf(eta, scale=0.5)
        logp_t=t.logpdf(nu,self.d)


        log_joint=logp_exp+logp_t -0.5*(np.log(self.beta_r)+np.log(r-epsilon-self.alpha_r+1e-6))+prior
        #log_joint=np.where(np.isnan(log_joint),-1e10,log_joint)
        #print(log_joint)
        return log_joint
        
    def sample(self, sample_num:int, r, exp_scale=1, resample_thre=0.5):
        self.ESS_list = []
        self.sample_num = sample_num
        
        samples = np.zeros((sample_num,self.T))
        log_weights = np.zeros(sample_num)
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
            if np.isnan(log_density).sum()+np.isnan(eps).sum() + np.isnan( log_weights).sum()>0:
                print(f"NaN encountered in sampling steps{i}")
                print(f"Parameters:{self.params}")


            
            r_past=rr
            samples[:,i]=eps
            log_weights=log_weights-np.min(log_weights)
            weights=np.exp(log_weights)
            #print("weights:",weights)
            weights=weights/weights.sum()
            log_weights=np.log(weights)
            
            ESS = 1/np.sum(np.power(weights, 2))
            self.ESS_list.append(ESS)
            if ESS < sample_num:
                samples[:,:i] = self.resample(samples[:,:i], weights)
                weights = np.ones(sample_num)/sample_num
                log_weights=np.zeros(sample_num)-np.log(sample_num)
                eps=samples[:,i]
            

        return samples, log_weights
    
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
        outputs = self.model(inputs).reshape(-1)
        assert torch.tensor(eps_past).shape[0]==outputs.shape[0]
        #outputs=torch.tensor(0.4).expand(inputs.shape[0],)
        #print(outputs)
        base=torch.distributions.Exponential(1).sample((inputs.shape[0],)).reshape(-1)
        sample=torch.tensor(rr)-torch.tensor(self.alpha_r)-base*outputs
        return sample.detach().numpy(),(torch.distributions.Exponential(1).log_prob(base)-torch.log(outputs)).detach().numpy()
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
    T=20
    r=np.load("./r.npy")
    params=(0.2, 0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5)
    #self.alpha_r, self.beta_r, self.d, self.alpha_u, self.beta_u, self.gamma, self.theta, self._lambda
    sampler=TEST_SAMPLER(T,params)
    samples,weights=sampler.sample(100000,r,exp_scale=0.5)
    print(sampler.ESS_list)
    print(r.shape,samples.shape,weights.shape)
    # print(samples)
    # print(weights)
