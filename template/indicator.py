import math
import torch

def log(x):
    if type(x) in (list,tuple):
        return [log(k) for k in x]
    if x<0:return math.nan
    if x==0:return float("-inf")
    return math.log(x)

class Indicator:
    def __init__(self):
        pass
    def reset(self):
        '''
        reset variable values
        '''
        raise NotImplementedError
    
    def add(self,out,ref):
        raise NotImplementedError
    @property
    def value(self):
        raise NotImplementedError