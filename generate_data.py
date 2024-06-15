import numpy as np
from scipy.stats import t, expon, norm

class Data_generator:
    def __init__(self, _alpha_r=0, _beta_r=1, _d=1, _alpha_u=0, _beta_u=1, _gamma=1, _theta=0, _lambda=1):
        self.alpha_r = _alpha_r
        self.beta_r = _beta_r
        self.d = _d
        self.alpha_u = _alpha_u
        self.beta_u = _beta_u
        self.gamma = _gamma
        self.theta = _theta
        self._lambda = _lambda

    def gen_data(self, n=1000, T=100):
        r=np.zeros((T,n)) #r yield
        eps=np.zeros(n)
        u=np.zeros(n)
        for i in range(T):
            u = self.alpha_u+self.beta_u*u+self.gamma*eps**2+self.theta*np.where(eps<0,eps**2,0)+norm.rvs(scale=0.5,size=n)
            eps=np.random.standard_t(df=self.d, size=n)*np.sqrt(u)
            r[i]=self.alpha_r+self.beta_r*u+eps
        return r
    
    def gen_data_full(self, n=1000, T=100):
        r=np.zeros((T,n)) #r yield
        eps_list=np.zeros((T,n))
        u_list=np.zeros((T,n))
        u=np.zeros(n)
        eps=np.zeros(n)
        for i in range(T):
            u = self.alpha_u+self.beta_u*u+self.gamma*eps**2+self.theta*np.where(eps<0,eps**2,0)+norm.rvs(scale=0.5,size=n)
            eps=np.random.standard_t(df=self.d, size=n)*np.sqrt(u)
            u_list[i]=u
            eps_list[i]=eps
            r[i]=self.alpha_r+self.beta_r*u+eps
        return r, u_list, eps_list

if __name__=='__main__':
    T_truth=1000
    T=300
    DG= Data_generator(0.2, 0.2, 6.0, 1, 0.4, 0.1, 0.02, 2.5)
    #DG = Data_generator(_alpha_r=2, _beta_r=1, _d=6, _alpha_u=4, _beta_u=0, _gamma=0, _theta=0, _lambda=0)
    r_data,u_data,eps_data=DG.gen_data_full(1, T_truth)
    print(r_data.shape)
    np.save('./data/r.npy',r_data[:T])
    np.save('./data/eps_truth.npy',eps_data)
    np.save('./data/u_truth.npy',u_data)
    np.save('./data/r_truth.npy',r_data)
    
#self.alpha_r, self.beta_r, self.d, self.alpha_u, self.beta_u,self.gamma, self.theta, self._lambda
#DG = Data_generator(0.2, 0.2, 6.0, 0.6, 0.4, 0.1, 0.02, 2.5)
#print(DG.gen_data_full(5, 5))    
#Test texts

#####################
#####################
##
##
#