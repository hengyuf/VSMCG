import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, expon, norm
    

class GAUSSIAN_SAMPLER:
    """test sampler"""
    ESS_list = []
    sample_num = 1
    def __init__(self, T, params):
        self.alpha_r, self.beta_r, self.d, self.alpha_u, self.beta_u,self.gamma, self.theta, self._lambda = params
        self.params = params
        self.T = T 

    def log_likelihood_update(self,epsilon,r,epsilon_past,r_past):
        ''' Compute log joint likelihood of l=log p(eps_t,r_t|eps_{t-1},r_{t-1})'''
        #Input:  epsilon  (n,) epsilon_past  (n,)  r (1,) r_past (1,)
        #Output: log_prob (n,)

        prior=0 #do we need prior on r_0, eps_0?

        ''' Calculate u, w, nu and eta'''
        #u=(r-epsilon-self.alpha_r)/self.beta_r
        u_past= (r_past-epsilon_past-self.alpha_r)/self.beta_r

        w=self.alpha_u+self.beta_u*u_past+(self.gamma+self.theta*(epsilon_past<0))*(epsilon_past**2)

        nu=np.sqrt(self.beta_r/(r-epsilon-self.alpha_r))*epsilon
        eta=(r-epsilon-self.alpha_r)/self.beta_r-w
        #eta=np.maximum(eta,1e-7)
        #print(f"eps:{epsilon[:10]}\n eps_past:{epsilon_past[:10]}\n r:{r} r_past:{r_past} etamin:{eta.min()}\n w:{w[:10]}, nu:{nu[:10]}\n eta:{eta[:10]}\n")
        #print(eta.min())
        #eta=(eta>=0)*eta+(eta<=0)*1e-7
        #assert eta.min()>=0 #eta should follow exponential distribution

        logp_exp=norm.logpdf(eta, scale=1/self._lambda)
        logp_t=t.logpdf(nu,self.d)


        log_joint=logp_exp+logp_t -0.5*(np.log(self.beta_r)+np.log(r-epsilon-self.alpha_r))+prior
        #print(log_joint)
        return log_joint
        
    def naive_sample(self, sample_num:int, r, norm_scale=0.5, norm_mean=1, resample_thre=0.2, seed=0):
        self.ESS_list = []
        self.sample_num = sample_num
        np.random.seed(seed)
        
        samples = np.zeros((sample_num,self.T))
        weights_full = np.zeros((sample_num,self.T))
        log_weights = np.ones(sample_num)
        eps=np.zeros(sample_num)
        r_past=self.alpha_r
        for i in range(self.T): 
            rr=r[i]
            eps_past=eps.copy()
            u_past= (r_past-eps_past-self.alpha_r)/self.beta_r
            w=self.alpha_u+self.beta_u*u_past+(self.gamma+self.theta*(eps_past<0))*(eps_past**2)
            if type(norm_scale)==list:
                temp_norm_scale = norm_scale[i]
            else: 
                temp_norm_scale = norm_scale
            if type(norm_mean)==list:
                temp_norm_mean = norm_mean[i]
            else: 
                temp_norm_mean = norm_mean
            eps = self.policy(eps_past, rr, w, temp_norm_scale, temp_norm_mean)
            log_weights += self.log_likelihood_update(eps,rr,eps_past,r_past)-self.log_policy_density(eps, rr, w, temp_norm_scale, temp_norm_mean)
            r_past=rr
            samples[:,i]=eps
            weights=np.exp(log_weights)
            weights=weights/weights.sum()
            
            ESS = 1/np.sum(np.power(weights, 2))
            self.ESS_list.append(ESS)
            if ESS < resample_thre*sample_num:
                samples[:,i] = self.resample(samples[:,i], weights)
                weights = np.ones(sample_num)/sample_num
                log_weights=np.zeros(sample_num)
            weights_full[:, i] = weights
        return samples, weights_full

    def sample(self, sample_num:int, r, norm_scale=0.5, norm_mean=1, resample_thre=0.2, max_iterations=5, seed=0):
        samples, weights_full = self.naive_sample(sample_num, r, norm_scale=norm_scale, norm_mean=norm_mean, resample_thre=resample_thre, seed=0)
        for _ in range(max_iterations):
            norm_mean_list = []
            norm_scale_list = []
            for i in range(self.T):
                mu = np.sum(weights_full[:,i]*samples[:,i])
                norm_mean_list.append(r[i]-self.params[0]-mu)
                var = np.sum(weights_full[:,i]*np.power(samples[:,i]-mu, 2))
                norm_scale_list.append(np.sqrt(var))
            samples, weights_full = self.naive_sample(sample_num, r, norm_scale=norm_scale_list, norm_mean=norm_mean_list, resample_thre=resample_thre, seed=0)
        return samples, weights_full
        
    
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
    
    def policy(self, eps_past, rr, w, norm_scale, norm_mean):
        #return rr-self.alpha_r-np.random.exponential(scale=exp_scale,size=self.sample_num) #a simple policy
        #return rr-self.alpha_r-(np.random.rand(self.sample_num)*0.8)**2
        return rr-self.alpha_r-np.abs(np.random.normal(scale=norm_scale,size=self.sample_num)+norm_mean) #a simple policy
    
    def log_policy_density(self, eps, rr, w, norm_scale, norm_mean):
        #return expon.logpdf(rr-self.alpha_r-eps, scale=exp_scale)
        #return np.log(0.8-(rr-self.alpha_r-eps))
        return np.log(norm.pdf(rr-self.alpha_r-eps-norm_mean, scale=norm_scale)+norm.pdf(-rr+self.alpha_r+eps-norm_mean, scale=norm_scale))
    

if __name__=="__main__":
    T=100
    r=np.load("./r.npy")
    print(r.shape)
    params=(0.2, 0.2, 6.0, 1.0, 0.4, 0.1, 0.02, 2.5)
    sampler=GAUSSIAN_SAMPLER(T,params)
    samples,weights=sampler.sample(10000,r,resample_thre=0.2)
    sampler.plot_ESS()
    print(r.shape,samples.shape,weights.shape)
    print(samples)
    index = np.random.choice(list(range(len(weights))), p=weights[:,-1], size=(len(weights)))
    unique_val=[]
    for t in range(T):
        unique_val.append(np.unique(samples[:,t][index]).shape[0])
    

    # plt.hist((samples[:,i])[index], density=True, bins=40, label="sampled")
    # plt.legend()
    # plt.show()
    print(unique_val)
    plt.plot(unique_val)
    plt.title("Unique values")
    plt.xlabel("t")
    plt.show()
    #print(weights)