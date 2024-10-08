import torch
from typing import NamedTuple,Dict,Tuple,Optional,Union,List

    
class Prefix(NamedTuple):
    '''
    prefix while searching

    prefix: (batch_size, num_steps, ) or (num_steps, )
    prob  : (batch_size, ) or (, )
    '''
    prefix:Union[torch.Tensor, List[int]]=[]
    prob  :Union[torch.Tensor, float]=0.
    def __lt__(self, prefix) -> bool:
        return self.prob<prefix.prob
    
    def unbatchify(self): # self.prefix and prob must be Tensor
        if isinstance(self.prob,float):
            return [Prefix(self.prefix[0].tolist(),self.prob)]
        return [Prefix(self.prefix[i].tolist(), self.prob[i].item())\
                for i in range(self.prob.size(0))]
    

class EndedPrefixes(NamedTuple):
    '''
    store results FOR ONE SAMPLE!

    prefixes: 
        unbatchfied prefixes: 
            prefix: (num_steps, )
            prob  : (, )
    '''
    prefixes:List[Prefix]=[]
    def append(self,ended_prefix:Prefix):
        self.prefixes.append(ended_prefix)
        
    def extend(self,ended_prefixes:List[Prefix]):
        self.prefixes.extend(ended_prefixes)
    
    @property
    def prefix(self):
        return [p.prefix for p in self.prefixes]

    def __iter__(self):
        return iter(self.prefixes)

    def sorted(self):
        self.prefixes.sort(reverse=True)
        return self
    

if __name__=="__main__":
    import sys,os
    sys.path.append(os.path.dirname(__file__))
    prefixes=EndedPrefixes([])
    print(prefixes.prefix)
    prefixes.extend(Prefix(torch.randn(0,2),torch.randn(0,1)).unbatchify())
    print(prefixes.prefix)
