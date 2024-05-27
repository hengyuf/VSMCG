import numpy as np
from scipy.stats import t, expon
class TEST_SAMPLER:
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
        #print(f"eps:{epsilon[:10]}\n eps_past:{epsilon_past[:10]}\n r:{r} r_past:{r_past} etamin:{eta.min()}\n w:{w[:10]}, nu:{nu[:10]}\n eta:{eta[:10]}\n")
        #assert eta.min()>0 #eta should follow exponential distribution
        logp_exp=expon.logpdf(eta, scale=1/self._lambda)
        logp_t=t.logpdf(nu,self.d)


        log_joint=logp_exp+logp_t -0.5*(np.log(self.beta_r)+np.log(r-epsilon-self.alpha_r))+prior
        #print(log_joint)
        return log_joint
        
    def sample(self, sample_num:int, r, exp_scale=1, resample_thre=0.2):
        self.ESS_list = []
        self.sample_num = sample_num
        
        samples = np.zeros((sample_num,self.T))
        log_weights = np.ones(sample_num)
        eps=np.zeros(sample_num)
        r_past=self.alpha_r
        for i in range(self.T): 
            rr=r[i]
            eps_past=eps.copy()
            eps = self.policy(eps_past, rr, r_past, exp_scale)
            log_weights += self.log_likelihood_update(eps,rr,eps_past,r_past)-self.log_policy_density(eps, eps_past, rr, r_past, exp_scale)
            r_past=rr
            samples[:,i]=eps
            weights=np.exp(log_weights)
            weights=weights/weights.sum()
            
            ESS = 1/np.sum(np.power(weights, 2))
            self.ESS_list.append(ESS)
            if ESS < resample_thre*sample_num:
                samples[:,i] = self.resample(samples[:,i], weights)
                weights = np.ones(sample_num)
                log_weights=np.zeros(sample_num)
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
    
    def policy(self, et1:np.array, rt, rt1, exp_scale):
        return rt-self.alpha_r-np.random.exponential(scale=exp_scale,size=self.sample_num) #a simple policy
    
    def log_policy_density(self, et:np.array, et1:np.array, rt, rt1, exp_scale):
        return expon.logpdf(rt- self.alpha_r - et, scale=exp_scale)

r=np.load("Bayes_temp/r.npy")
params=(0.2, 0.2, 6.0, 0.6, 0.4, 0.1, 0.02, 2.5)
sampler=TEST_SAMPLER(5,params)
samples,weights=sampler.sample(1000,r,1)
sampler.plot_ESS()
print(r.shape,samples.shape,weights.shape)