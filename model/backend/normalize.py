from torch import nn


class Residual(nn.Module):
    def __init__(self,f,scale,dropout):
        super().__init__()
        self.f = f
        self.scale=scale
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,*args,**kwargs):
        ret=self.f(x,*args,**kwargs)
        if type(ret) is tuple:
            out,*fout=ret
        else:out=ret

        out=self.scale*self.dropout(out)

        if type(ret) is tuple:
            return x+out,*fout
        return out

class PreNorm(nn.Module):
    def __init__(self,norm_shape,f=nn.Identity(),scale=1.,dropout=0):
        super().__init__()
        self.ln=nn.LayerNorm(norm_shape,eps=1e-12)
        self.f = f
        self.scale=scale
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,*args,**kwargs):
        out=self.ln(x)
        ret=self.f(out,*args,**kwargs)
        if type(ret) is tuple:
            out,*fout=ret
        else:out=ret
        
        out=self.scale*self.dropout(out) 

        if type(ret) is tuple:
            return x+out,*fout
        return x+out
    
class AddNorm(nn.Module):
    def __init__(self,norm_shape,f=nn.Identity(),scale=1.,dropout=0):
        super().__init__()
        self.ln=nn.LayerNorm(norm_shape,eps=1e-12)
        self.f = f
        self.scale=scale
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,*args,**kwargs):
        ret=self.f(x,*args,**kwargs)
        if type(ret) is tuple:
            out,*fout=ret
        else:out=ret

        out=self.scale*self.dropout(out)
        out=self.ln(out+x)
        
        if type(ret) is tuple:
            return out,*fout
        return out



# class A(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.m=nn.Linear(5,5)
#         self.n=nn.Linear(5,5)
#     def forward(self,x,y):
#         return self.m(x),self.n(x)+y

if __name__=="__main__":
    # a=PreNorm(5,A())
    # import torch
    # t=torch.randn(2,3,5)
    # # y=torch.randn(2,3,5)
    # m,n=a(t,t)
    # print(m,n)
    # print(a(t,t))
    # print(a(t,y).shape)

    pass
    


