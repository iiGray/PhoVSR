import torch
from typing import Optional,Union,NamedTuple

class Vocab:
    def __init__(self,char_list):
        self.char_list=char_list
        self.char_dict={char_list[i]:i for i in range(len(char_list))}
        assert \
            "<sos>" in self.char_dict and \
                "<eos>" in self.char_dict and \
                    "<pad>" in self.char_dict and \
                        "<unk>" in self.char_dict
            
    def __len__(self):
        return len(self.char_list)
    def __getitem__(self,w):
        if isinstance(w,int):
            return self.char_list[w]
        if isinstance(w,str):
            return self.char_dict[w] if w in self.char_dict else self.char_dict["<unk>"]
        return [self.__getitem__(k) for k in w]
    


class State:
    '''
    It can be used for any binary situation,
    where the first represents the output features,
          the second represents the valid_len of the features

    enc or dec output
    feats: (b, t) or (b, t, f)
    valen: valid_len (b, )
    '''
    def __init__(self,
                 feats:Optional[torch.Tensor] =None,
                 valen:Union[torch.Tensor,int]=None
                 ):
        self.feats=feats
        self.valen=torch.full((feats.size(0),),
                              feats.size(1),
                              device=feats.device).long() \
            if valen is None \
                else torch.tensor([valen],
                                  device=feats.device) \
                    if isinstance(valen,int) \
                        else valen

        self.device=feats.device
        self.dtype=feats.dtype
    def __len__(self):return self.feats.size(0)

    def __bool__(self):return 0 not in self.feats.shape

    def __iter__(self):
        '''
        not for iter ,but for (*State() ),
        if need itering, use for s,l in zip(*State())
        '''
        return iter((self.feats,self.valen))
    
    def __repr__(self):
        return f"State(feats={self.feats.shape}, valen={self.valen.shape})"

    def to(self,*args,**kwargs):
        self.feats=self.feats.to(*args,**kwargs)
        self.valen=self.valen.to(*args,**kwargs)
        return self
    

    def set(self,state):
        
        self.feats=state.feats
        self.valen=state.valen

        return self

    def select_(self,ids):
        '''
        designed for ctc:
        '''
        assert self.feats.ndim==3,\
            f"feats shape: {self.feats.shape }, ids shape:{ids.shape} SelectFaile."

        b, t, *args=self.feats.shape
        '''
        keep different time steps in one batch in same
        '''
        self.feats=self.feats[torch.arange(b,device=self.feats.device)[:,None,None],
                              torch.arange(t,device=self.feats.device)[None,:,None],
                              ids]
        return self

    def select(self,ids):
        b, t, *args=self.feats.shape

        assert self.feats.ndim==3,\
            f"feats shape: {self.feats.shape }, ids shape:{ids.shape} SelectFaile."

        return State(
            self.feats[torch.arange(b,device=self.feats.device)[:,None,None],
                       torch.arange(t,device=self.feats.device)[None,:,None],
                       ids],
            self.valen
        )

    def setitem(self,key):
        '''
        the same as __getitem__, but while change self value
        '''
        self.feats=self.feats[key]
        valen=self.valen[key[0] if isinstance(key,tuple) else key]
        if isinstance(key,tuple) and len(key)>1:
            valen=valen[:,None]>torch.arange(valen.max())[None,:]
            valen=valen[:,key[1]].sum(-1)
        self.valen=valen
        return self

    def __getitem__(self,key):
        '''
        key shape:
        (batch_select_dim,) or
        tuple((batch_select_dim,),)  or
        (slice(),slice())

        if key shape is    : (batch_select_dim, 1)
        then feats will be : (batch_select_dim, 1, time_dim, feats_dim)
             valen will be : (batch_select_dim, 1)
        '''
        feats=self.feats[key]
        valen=self.valen[key[0] if isinstance(key,tuple) else key]
        if isinstance(key,tuple) and len(key)>1:
            valen=valen[:,None]>torch.arange(valen.max())[None,:]
            valen=valen[:,key[1]].sum(-1)
        return State(feats,valen)
    


    def repeat_(self,repeats):
        self.feats=torch.repeat_interleave(self.feats,repeats=repeats,dim=0)
        self.valen=torch.repeat_interleave(self.valen,repeats=repeats,dim=0)
        return self
    
    def concat_(self,state):
        assert state.feats.ndim<=2 and \
            (self.valen.numel()==0 or \
             self.valen.min()==self.valen.max().item()),\
            f"Concat {self.feats.shape} with {state.feats.shape} Failed !"
        
        self.feats=torch.concat([self.feats,state.feats],dim=-1)
        self.valen+=state.valen
        return self
    
    def concat(self,state):
        assert state.feats.ndim<=2 and \
            (self.valen.numel()==0 or \
             self.valen.min()==self.valen.max().item()),\
            f"Concat {self.feats.shape} with {state.feats.shape} Failed !"
        
        return State(torch.concat([self.feats,state.feats],dim=-1),
                     self.valen + state.valen)
    
    

class EvalModule:
    '''
    Decoders have to inherit it to gain its methods
    
    Use the method 'scorer' to gain a decorated module, which is able to 
    compute probabilities relying on eState by using method 'scoring' directly
    '''
    
    def scorer(self,eState:State,weight=1.0):
        return Scorer(self,self.initState(eState),weight)
    
    def initState(self,eState:State, *args, **kwargs) -> State:
        '''
        init scorer's state before scoring, omit allowed

        but have to return final state
        '''
        return eState
    
    def preProcess(self,*args,**kwargs):
        '''
        use if necessary before THE FIRST scoring, omit allowed
        '''
        return 

    def postProcess(self, *args, **kwargs):
        '''
        use if necessary after EVERY scoring, omit allowed
        '''
        return 
    
    def selectNext(self, bk_pref, bk_pred, *args, **kwargs):
        '''
        select non-raw State of each scorer(rawState is updated by self.rState.setitem(bk_pref))
        '''
        return
    
    def scoring(self,eState:State,dState:State,*args,**kwargs) -> torch.Tensor:
        '''
        return: next character pred probabilities: 
                shape (batch_size=1, vocab_size)
        '''
        raise NotImplementedError

class Scorer(NamedTuple):
    '''
    Decorate a decode model to gain weight and score

    Normally not be used directly, but to be used via the method 'scorer' of EvalModule
    '''    
    model:EvalModule
    state:State 
    weight:float=1.0
    # score:Union[torch.Tensor,float]=0

    def initState(self, eState: State,*args, **kwargs):
        return self.model.initState(eState, *args, **kwargs)

    def preProcess(self, *args, **kwargs):
        return self.model.preProcess(*args, **kwargs)

    def postProcess(self, *args, **kwargs):
        return self.model.postProcess(*args, **kwargs)

    def selectNext(self, bk_pref, bk_pred, *args, **kwargs):
        return self.model.selectNext(bk_pref, bk_pred, *args, **kwargs)

    def setState(self,newState: State):
        self.state.set(newState)

    def scoring(self,dState:State=None,*args,**kwargs):
        # assert self.state.feats is not None
        # assert isinstance(self.model,EvalModule)
        return self.model.scoring(self.state,dState,*args,**kwargs)
    
    def weighted_scoring(self,dState:State=None,*args,**kwargs):
        return self.scoring(dState, *args, **kwargs)*self.weight
    
    @property
    def device(self):
        return self.state.device

