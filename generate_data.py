import numpy as np

class PARAMS:
    def __init__(self, _alpha_r=0, _beta_r=1, _d=1, _alpha_u=0, _beta_u=1, _gamma=1, _theta=0, __lambda=1):
        self.alpha_r = _alpha_r
        self.beta_r = _beta_r
        self.d = _d
        self.alpha_u = _alpha_u
        self.beta_u = _beta_u
        self.gamma = _gamma
        self.theta = _theta
        self._lambda = __lambda

def gen_data(params, n=1000, T=100):
    r=np.zeros((T,n)) #r yield
    eps=np.zeros(n)
    u=np.zeros(n)
    for i in range(T):
        u = params.alpha_u+params.beta_u*u+params.gamma*eps**2+params.theta*np.where(eps<0,eps**2,0)+np.random.exponential(scale=1/params._lambda,size=n)
        eps=np.random.standard_t(df=params.d, size=n)*np.sqrt(u)
        r[i]=params.alpha_r+params.beta_r*u+eps
    return r
    
params = PARAMS(0.2, 0.2, 6.0, 0.6, 0.4, 0.1, 0.02, 2.5)
gen_data(params, 5, 5)
