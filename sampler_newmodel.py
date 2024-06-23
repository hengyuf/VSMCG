import numpy as np
from scipy.stats import t, expon, norm
import matplotlib.pyplot as plt
import torch.distributions as tdist
import pdb
import torch
from torch.distributions import Normal
import torch.nn as nn
from TruncatedNormal import TruncatedNormal
from VIScaler import VIScaler
import logging

# Configure logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True
log = logging.getLogger('Sampler Logger')
log.setLevel(logging.DEBUG)

# import line_profiler
# profile = line_profiler.LineProfiler()

def param_to_input(r,epsilon_past,r_past,alpha_r, beta_r, d, alpha_u, beta_u,gamma, theta, _lambda):
    #epsilon_past is of shape (n,) everying else scalers
    n=epsilon_past.shape[0]
    output=torch.zeros(n,3)
    u_past=(r_past-alpha_r-epsilon_past)/beta_r
    output[:,0]=alpha_u+beta_u*u_past+gamma*epsilon_past**2+theta*(epsilon_past<0)*epsilon_past**2
    output[:,1]=r-alpha_r
    output[:,2]=beta_r

    return output



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
    def __init__(self, T, params,path="VIScaler_test1_99.pth",debug=False):
        self.alpha_r, self.beta_r, self.d, self.alpha_u, self.beta_u,self.gamma, self.theta, self._lambda = params
        # print(params)
        self.params = params
        self.T = T
                
        self.model=VIScaler(hidden_size=16)
        self.model=torch.load(path).to('cpu')
        self.model.eval()
        self.base_dist=tdist.Gamma(1.5,0.75)
        self.base_dist2=HalfNormal(1)

        self.debug=debug

    def update_params(self,params):
        self.params = params
        self.alpha_r, self.beta_r, self.d, self.alpha_u, self.beta_u,self.gamma, self.theta, self._lambda = params

    def log_likelihood_update(self,epsilon,r,epsilon_past,r_past):
        ''' Compute log joint likelihood of l=log p(eps_t,r_t|eps_{t-1},r_{t-1})'''
        #Input:  epsilon  (n,) epsilon_past  (n,)  r (1,) r_past (1,)
        #Output: log_prob (n,)

        prior=0 #do we need prior on r_0, eps_0?

        ''' Calculate u, w, nu and eta'''
        #u=(r-epsilon-self.alpha_r)/self.beta_r
        u_past= (r_past-epsilon_past-self.alpha_r)/self.beta_r

        w=self.alpha_u+self.beta_u*u_past+(self.gamma+self.theta*(epsilon_past<0))*(epsilon_past**2)

        nu=np.sqrt(self.beta_r/(r-epsilon-self.alpha_r+1e-4))*epsilon
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
    
    def sample(self, sample_num:int, r, exp_scale=1, resample_thre=0.1, checklist=[], return_prob=False):
        self.ESS_list = []
        self.sample_num = sample_num
        
        samples = np.zeros((sample_num,self.T))
        log_weights = np.ones(sample_num)
        eps=np.zeros(sample_num)
        r_past=self.alpha_r
        log_truth_weights=np.ones(sample_num)
        for i in range(self.T): 
            # print(f"----------STEP{i}----------")
            #pdb.set_trace()
            rr=r[i]
            eps_past=eps.copy()
            u_past= (r_past-eps_past-self.alpha_r)/self.beta_r
            w=self.alpha_u+self.beta_u*u_past+(self.gamma+self.theta*(eps_past<0))*(eps_past**2)
            eps,log_density = self.policy(eps_past, rr, w, exp_scale,r_past)
            # print(f"step{i}\n eps:{eps}\n eps_past:{eps_past} \n rr:{rr} w:{w} exp_scale:{exp_scale}\n")

            prob_update=self.log_likelihood_update(eps,rr,eps_past,r_past)-log_density
            log_weights += prob_update      #self.log_policy_density(eps, rr, w, exp_scale,r_past)
            log_truth_weights+=prob_update

            samples[:,i]=eps
            weights=np.exp(log_weights)
            weights=weights/weights.sum()
            
            ESS = 1/np.sum(np.power(weights, 2))
            self.ESS_list.append(ESS)
            if self.debug and (ESS<0.1*sample_num*resample_thre or i in checklist or ESS<0.3*self.ESS_list[-min(2,i+1)]):

                log.debug(f"----Debug Information of Timestep {i}----")
                log.debug(f"ESS: {ESS} Last ESS: {self.ESS_list[-min(2,i+1)]}")
                log.debug(f"Weights min: {min(weights)} max: {max(weights)}")
                log.debug(f"rpast: {r_past} rcurrent: {rr}")
                log.debug(f"Params: {self.params}")
                

                # #Plot Weights
                # plt.figure()
                # num_bins = 42
                # log_bins = np.logspace(np.log10(1e-14), np.log10(1), num_bins)
                # # plt.hist(weights, bins=log_bins, label=f"Histogram of Weights At Time {i}")
                # plt.hist(eps, bins=30, label=f"Histogram of Epsilon At Time {i}")
                # plt.yscale('log')
                # plt.xscale('log')
                # #plt.savefig(f'./debug_figs/Debug-Weights-t={i}_alpha_r={self.alpha_r}.png')
                # plt.show()
                # #plt.close()

                #Info About Largest
                index=np.argmax(weights)
                output1,output2,inputs=self.policy(eps_past, rr, w, exp_scale,r_past,getoutput=True)

                log.debug(f"Index of largest point: {index}")
                
               
                log.debug(f"Index of largest point: {index}")
                log.debug(f"eps[index]: {eps[index]} eps_min: {min(eps)} eps_max: {max(eps)}")
                log.debug(f"eps_past[index]: {eps_past[index]}  eps_min: {min(eps)} eps_max: {max(eps)}")
                log.debug(f"output1[index]: {output1[index]}")
                log.debug(f"output2[index]: {output2[index]}")
                log.debug(f"input[index]: {inputs[index]}")
                log.debug(f"-----------------------------------------")
            if ESS < sample_num*resample_thre:
                samples[:,:i],index = self.resample(samples[:,:i], weights)
                weights = np.ones(sample_num)/sample_num
                log_weights=np.zeros(sample_num)
                eps=samples[:,i]
                log_truth_weights=log_truth_weights[index]
            r_past=rr

        if return_prob:
            return samples, weights, log_truth_weights
        return samples, weights
    
    def plot_ESS(self, y_high=0, title=""):
        if y_high == 0:
            y_high = self.sample_num
        plt.figure()
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
        return samples[index],index


    def policy(self, eps_past, rr, w, exp_scale,r_past,getoutput=False):
        #getoutput=True and the function will change to return model outputs

        # print("exponentials:",expon.rvs(scale=exp_scale,size=self.sample_num))
        # print("values:",rr,self.alpha_r,self.beta_r*w,self.beta_r*np.random.exponential(scale=exp_scale,size=self.sample_num))
        #print("generate eps:",rr-self.alpha_r-self.beta_r*w-self.beta_r*np.random.exponential(scale=exp_scale,size=self.sample_num))
        inputs=param_to_input(torch.tensor(rr),torch.tensor(eps_past),torch.tensor(r_past),torch.tensor(self.alpha_r),torch.tensor(self.beta_r),
                              torch.tensor(self.d),torch.tensor(self.alpha_u),torch.tensor(self.beta_u),torch.tensor(self.gamma),
                              torch.tensor(self.theta),torch.tensor(self._lambda))
        outputs,outputs2,outputs3,outputs4 = self.model(inputs)
        if getoutput:
            return outputs,outputs2,inputs
        outputs,outputs2,outputs3,outputs4 =outputs.reshape(-1),outputs2.reshape(-1),outputs3.reshape(-1),outputs4.reshape(-1)
        assert eps_past.shape[0]==outputs.shape[0]
        assert outputs.min()>=0
        assert outputs2.min()>=0
        #outputs=torch.tensor(0.4).expand(inputs.shape[0],)
        # base1=self.base_dist.sample((inputs.shape[0],)).reshape(-1)
        # base2=self.base_dist2.sample((inputs.shape[0],)).reshape(-1)
        # random_tensor = torch.rand_like(outputs4)

        # base = base1#torch.where(random_tensor < outputs4, base1, base2)
        # baselogprob=self.base_dist.log_prob(base) #torch.log(outputs4*torch.exp(self.base_dist.log_prob(base))+(1-outputs4)*torch.exp(self.base_dist2.log_prob(base)))
        # #print(outputs2)
        # modifiedbase=outputs*base+outputs2*base**1.5+outputs3*base**0.5
        # jacobian=outputs+1.5*base**0.5*outputs2+0.5*outputs3*base**(-0.5)

        base_dist=TruncatedNormal(loc=outputs,scale=outputs2,a=1e-4,b=100)

        #print(batch[1].shape)

        modifiedbase= base_dist.rsample(sample_shape=torch.ones(1).shape).reshape(-1)
        #print(modifiedbase.shape)
        # base1=base_dist.sample((batch_data.shape[0],)).reshape(-1)
        # #base2=base_dist2.sample((batch_data.shape[0],)).reshape(-1)
        # random_tensor = torch.rand_like(outputs4)

        # base = base1#torch.where(random_tensor < outputs4, base1, base2)

        # modifiedbase=outputs*base+outputs2*base**1.5+outputs3*base**0.5
        # assert modifiedbase.min()>0
        # jacobian=outputs+1.5*base**0.5*outputs2+0.5*outputs3*base**(-0.5)

        # #modifiedbase=outputs*base
        # #jacobian=outputs
        # #print("modified base1",modifiedbase)

        # print("modified base:",modifiedbase.min(),modifiedbase.max())
        # print("logprob:",logprob.min(),logprob.max())
        # print(f"batch:{batch[1].max()} {batch[1].min()} {batch[4].max()} {batch[4].min()}")
        
        baselogprob=base_dist.log_prob(modifiedbase)  #torch.log(outputs4*torch.exp(base_dist.log_prob(base))+(1-outputs4)*torch.exp(base_dist2.log_prob(base)))

        sample=torch.tensor(rr)-torch.tensor(self.alpha_r)-modifiedbase
        return sample.detach().numpy(),baselogprob.detach().numpy()   #sample.detach().numpy(),(baselogprob-torch.log(jacobian)).detach().numpy()
        #return rr-self.alpha_r-expon.rvs(scale=exp_scale,size=self.sample_num)
    #np.random.exponential(scale=exp_scale,size=self.sample_num) #a simple policy
    
    def policy_old(self, eps_past, rr, w, exp_scale,r_past):
        eps=rr-self.alpha_r-expon.rvs(scale=exp_scale,size=self.sample_num)
        return eps, expon.logpdf((rr-self.alpha_r-eps),scale=exp_scale)
    
    def log_policy_density(self, eps,eps_past, rr,r_past):
        inputs=param_to_input(torch.tensor(rr),torch.tensor(eps_past),torch.tensor(r_past),torch.tensor(self.alpha_r),torch.tensor(self.beta_r),
                              torch.tensor(self.d),torch.tensor(self.alpha_u),torch.tensor(self.beta_u),torch.tensor(self.gamma),
                              torch.tensor(self.theta),torch.tensor(self._lambda))
        outputs,outputs2,outputs3,outputs4 = self.model(inputs)
        outputs,outputs2,outputs3,outputs4 =outputs.reshape(-1),outputs2.reshape(-1),outputs3.reshape(-1),outputs4.reshape(-1)
        assert torch.tensor(eps_past).shape[0]==outputs.shape[0]

        base_dist=TruncatedNormal(loc=outputs,scale=outputs2,a=1e-4,b=100)

        #print(batch[1].shape)

        modifiedbase= torch.tensor(rr)-torch.tensor(self.alpha_r)-eps

        assert modifiedbase.min()>0

        
        baselogprob=base_dist.log_prob(modifiedbase)  #torch.log(outputs4*torch.exp(base_dist.log_prob(base))+(1-outputs4)*torch.exp(base_dist2.log_prob(base)))

        return baselogprob








if __name__=="__main__":
    T=300
    r=np.load("./data/r.npy")
    print(r.shape)
    params=(0.2, 0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5)
    sampler=TEST_SAMPLER(T,params,path='./tmppth/VIScaler_test1_888.pth')
    samples,weights, log_prob=sampler.sample(10000,r,resample_thre=0.2,return_prob=True)
    sampler.plot_ESS()
    # #print(r.shape,samples.shape,weights.shape)
    # print(samples)
    # i=99
    # index = np.random.choice(list(range(len(weights))), p=weights, size=(len(weights)))
    # plt.hist((samples[:,i])[index], density=True, bins=40, label="sampled")
    # plt.legend()
    # plt.show()

    # unique_val=[]
    # for t in range(T):
    #     unique_val.append(np.unique(samples[:,t][index]).shape[0])
    

    # plt.hist((samples[:,i])[index], density=True, bins=40, label="sampled")
    # plt.legend()
    # # plt.show()
    # print(unique_val)
    # plt.plot(unique_val)
    # plt.title("Unique values")
    # plt.xlabel("t")
    # plt.show()
    #print(weights)
    # profile.print_stats()