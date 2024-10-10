from template.indicator import *

class Bleu(Indicator):
    def __init__(self,k=4,weights=(0.25,0.25,0.25,0.25)):
        assert k==len(weights)
        self.k=k
        self.weights=weights
        self.P=[0]*(k+1)
        self.L=[0]*(k+1)
    
    def reset(self):
        self.P=[0]*len(self.P)
        self.L=[0]*len(self.L)

    def add(self,out,ref):
        if type(out)!=list:
            out=out.tolist()
            ref=ref.tolist()
        
        self.L[0]+=len(ref)
        self.P[0]+=len(out)

        st=[set() for _ in range(5)]
        for i in range(4):
            for l in range(len(out)-i):
                r=l+i+1
                st[r-l].add(tuple(out[l:r]))

        for i in range(4):
            if(len(out)>i):self.L[i+1]+=len(out)-i
            has=0
            for l in range(len(ref)-i):
                r=l+i+1
                if(tuple(ref[l:r]) in st[r-l]):
                    has+=1
                if has>=len(out)-i:break
            self.P[i+1]+=has
    @property
    def value(self):
        BP=min(1,math.exp(1-(self.L[0]/self.P[0])))
        # logv=sum(log)
        logv=log([p/l for p,l in \
                      zip(self.P[1:],self.L[1:])])
        logsum=sum([v*w for v,w in zip(logv,self.weights)])
        
        return float(BP*math.exp(logsum))
    
class CER(Indicator):
    def __init__(self):
        self.error=0
        self.total=0
    
    def reset(self):
        self.error=self.total=0

    def add(self,
            out:list,
            ref:list
            ):
        '''
        out   :  (*, out size)   1 dim in all
        ref   :  (*, ref size)   1 dim in all

        return:  tuple(int,int) ,the first is error nums, the second is total nums
        '''
        cost=torch.zeros((len(out)+1,len(ref)+1))
        for i in range(len(out)+1):
            cost[i,0]=i
        for j in range(len(ref)+1):
            cost[0,j]=j

        for i in range(1,len(out)+1):
            for j in range(1,len(ref)+1):
                cost[i,j]+= 0 if out[i-1]==ref[j-1] else 1
                cost[i,j]+=min(
                    cost[i-1,j-1],
                    cost[i,j-1],
                    cost[i-1,j]
                )
        self.error+=cost[-1,-1].item()
        self.total+=len(ref)
    
    @property
    def value(self):return float(self.error/self.total)

