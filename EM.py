import numpy as np

class EM:
    def __init__(self,T) -> None:
        self.T=T
    def load_data(self,n):
        return np.random.randn(n), np.random.randn(n,self.T)

EM_sampler=EM(10)
a,b=EM_sampler.load_data(5)
print(a.shape,b.shape)