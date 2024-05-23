import numpy as np
def gen_data(alpha_r=0.2,beta_r=0.2,d=6.0,alpha_u=0.6,beta_u=0.4,gamma=0.1,theta=0.02, _lambda=2.5,n=1000,T=100):
    r=np.zeros((T,n)) #r yield
    eps=np.zeros(n)
    u=np.zeros(n)
    for i in range(T):
        u=alpha_u+beta_u*u+gamma*eps**2+theta*np.where(eps<0,eps**2,0)+np.random.exponential(scale=1/_lambda,size=n)
        eps=np.random.standard_t(df=d,size=n)*np.sqrt(u)
        r[i]=alpha_r+beta_r*u+eps
    return r







    