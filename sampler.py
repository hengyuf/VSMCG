import numpy as np
from scipy.stats import t, expon, norm
import matplotlib.pyplot as plt
import pdb
class TEST_SAMPLER:
    """test sampler"""
    ESS_list = []
    sample_num = 1
    def __init__(self, T, params):
        self.alpha_r, self.beta_r, self.d, self.alpha_u, self.beta_u,self.gamma, self.theta, self._lambda = params
        # print(params)
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
        
    def sample(self, sample_num:int, r, exp_scale=1, resample_thre=0.5):
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
            eps = self.policy(eps_past, rr, w, exp_scale)
            # print(f"step{i}\n eps:{eps}\n eps_past:{eps_past} \n rr:{rr} w:{w} exp_scale:{exp_scale}\n")
            log_weights += self.log_likelihood_update(eps,rr,eps_past,r_past)-self.log_policy_density(eps, rr, w, exp_scale)

            
            r_past=rr
            samples[:,i]=eps
            weights=np.exp(log_weights)
            weights=weights/weights.sum()
            
            ESS = 1/np.sum(np.power(weights, 2))
            self.ESS_list.append(ESS)
            if ESS < sample_num:
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
    
    def policy(self, eps_past, rr, w, exp_scale):
        # print("exponentials:",expon.rvs(scale=exp_scale,size=self.sample_num))
        # print("values:",rr,self.alpha_r,self.beta_r*w,self.beta_r*np.random.exponential(scale=exp_scale,size=self.sample_num))
        #print("generate eps:",rr-self.alpha_r-self.beta_r*w-self.beta_r*np.random.exponential(scale=exp_scale,size=self.sample_num))

        return rr-self.alpha_r-expon.rvs(scale=exp_scale,size=self.sample_num)
    #np.random.exponential(scale=exp_scale,size=self.sample_num) #a simple policy
    
    def log_policy_density(self, eps, rr, w, exp_scale):
        # print("exponential values:",(rr-self.alpha_r-self.beta_r*w-eps)/self.beta_r)
        # print("logpdf:",expon.logpdf((rr-self.alpha_r-self.beta_r*w-eps)/self.beta_r,scale=exp_scale))
        return expon.logpdf((rr-self.alpha_r-eps),scale=exp_scale)





if __name__=="__main__":
    r=np.load("./r.npy")
    params=(0.2, 0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5)
    sampler=TEST_SAMPLER(5,params)
    samples,weights=sampler.sample(10000,r,exp_scale=0.4)
    #sampler.plot_ESS()
    print(r.shape,samples.shape,weights.shape)
    print(samples)
    #print(weights)
