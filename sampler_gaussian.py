'''
This is code for a sampler with Gaussian policy.
The sampler updates parameters by simple estimations of empirical mean and variance.
Run this code to get Figures 2 and 4. 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
import time
    

class GAUSSIAN_SAMPLER:
    """test sampler"""
    def __init__(self, T, params):
        self.alpha_r, self.beta_r, self.d, self.alpha_u, self.beta_u,self.gamma, self.theta, self._lambda = params
        self.params = params
        self.T = T 

    def log_likelihood_update(self,epsilon,r,epsilon_past,r_past):
        ''' Compute log joint likelihood of l=log p(eps_t,r_t|eps_{t-1},r_{t-1})'''
        #Input:  epsilon  (n,) epsilon_past  (n,)  r (1,) r_past (1,)
        #Output: log_prob (n,)

        prior=0

        ''' Calculate u, w, nu and eta'''
        u_past=(r_past-epsilon_past-self.alpha_r)/self.beta_r
        w=self.alpha_u+self.beta_u*u_past+(self.gamma+self.theta*(epsilon_past<0))*(epsilon_past**2)
        nu=np.sqrt(self.beta_r/(r-epsilon-self.alpha_r))*epsilon
        eta=(r-epsilon-self.alpha_r)/self.beta_r-w

        logp_exp=norm.logpdf(eta, scale=1/self._lambda)
        logp_t=t.logpdf(nu,self.d)

        log_joint=logp_exp+logp_t -0.5*(np.log(self.beta_r)+np.log(r-epsilon-self.alpha_r))+prior
        return log_joint
        
    def naive_sample(self, sample_num:int, r, norm_scale=0.5, norm_mean=1, resample_thre=0.2, seed=0):
        '''basic sampler'''
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

    def sample(self, sample_num:int, r, norm_scale=0.5, norm_mean=1, resample_thre=0.2, max_iterations=5, seed=0, print_info=True, print_fig=False):
        '''iterates to improve sampler'''
        if print_info:
            print("-"*30)
            print("Sampling with Gaussian sampler")
            print("length", self.T, "sample num:", sample_num)
            print("max iterations:", max_iterations, "resample threshold:", resample_thre)
            print("params:", self.params)
        start_t = time.time()
        samples, weights_full = self.naive_sample(sample_num, r, norm_scale=norm_scale, norm_mean=norm_mean, resample_thre=resample_thre, seed=seed)
        if print_info:
            print("iterations:", "{}/{}".format("%2d"%0, "%2d"%max_iterations), "time: {}s".format("%3.3f"%(time.time()-start_t)))
        if print_fig: 
            self.plot_ESS(title="Iterations: 0")
        for _ in range(max_iterations):
            if print_info:
                print("iterations:", "{}/{}".format("%2d"%(_+1), "%2d"%max_iterations), "time: {}s".format("%3.3f"%(time.time()-start_t)))
            norm_mean_list = []
            norm_scale_list = []
            for i in range(self.T):
                mu = np.sum(weights_full[:,i]*samples[:,i])
                norm_mean_list.append(r[i]-self.params[0]-mu)
                var = np.sum(weights_full[:,i]*np.power(samples[:,i]-mu, 2))
                norm_scale_list.append(np.sqrt(var))
            samples, weights_full = self.naive_sample(sample_num, r, norm_scale=norm_scale_list, norm_mean=norm_mean_list, resample_thre=resample_thre, seed=seed)
            if print_fig and (_ in [0,2,9]): 
                self.plot_ESS(title="Iterations: "+str(_+1))
        if print_info:
            print("-"*30)
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
        plt.tight_layout(pad=0.2)
        plt.show()
        plt.clf()
    
    def resample(self, samples, weights):
        index = np.random.choice(list(range(len(weights))), p=weights, size=(len(weights)))
        return samples[index]
    
    def policy(self, eps_past, rr, w, norm_scale, norm_mean):
        return rr-self.alpha_r-np.abs(np.random.normal(scale=norm_scale,size=self.sample_num)+norm_mean) #a Gaussian policy
    
    def log_policy_density(self, eps, rr, w, norm_scale, norm_mean):
        return np.log(norm.pdf(rr-self.alpha_r-eps-norm_mean, scale=norm_scale)+norm.pdf(-rr+self.alpha_r+eps-norm_mean, scale=norm_scale))
    

def run_sampler_gaussian(print_info=False):
    T = 100
    r = np.load("./data/r.npy")
    eps_truth = np.load("./data/eps_truth.npy")
    params = (0.2, 0.2, 6.0, 1.0, 0.4, 0.1, 0.02, 2.5)
    sampler = GAUSSIAN_SAMPLER(T, params)
    samples, weights = sampler.sample(10000, r, resample_thre=0.2, max_iterations=10, seed=1, print_info=print_info, print_fig=True)
    
    for i in [24,49,74,99]:
        index = np.random.choice(list(range(len(weights))), p=weights[:, i], size=(len(weights)))
        plt.hist((samples[:,i])[index], density=True, bins=40, label="sampled")
        plt.axvline(x=r[i]-params[0],ls="--",c="C1",label=r"$r_t-\alpha_r$")
        plt.axvline(x=eps_truth[i],ls="--",c="C2",label=r"truth $\epsilon_t$")
        plt.title(f"$t={i+1}$")
        plt.legend()
        plt.tight_layout(pad=0.2)
        plt.show()
        plt.clf()

if __name__ == "__main__":
    run_sampler_gaussian(print_info=True)
