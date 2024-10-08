#!/usr/bin/env python
# coding: utf-8

import torch,collections,cv2
from torch import nn
import numpy as np,pandas as pd
import sys,time,os,shutil
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display

from collections import defaultdict
from typing import List,Dict,Union

'''Timer/Solution/Accumulator/Animator'''


class Timer:
    def __init__(self):
        self.times=[]
        self.start()
        self.begin=time.time()
    
    def restart(self):
        self.begin=time.time()
    @property
    def interval(self):
        ret=time.time()-self.begin
        return ret

    def start(self):
        self.s=time.time()
    def stop(self):
        return time.time()-self.s
        self.times.append(time.time()-self.s)
        return self.times[-1]
    def avg(self):
        return sum(self.times)/len(self.times)
    def sum(self):
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    
    def format(self,interval):
        units=["s","min","h","d","m","y"]
        scales,valid=[60,60,24,30,12,10],[]
        for i,scale in enumerate(scales):
            scales[i]=interval%scale
            interval//=scale
            if scales[i]:valid.append(i)
            if interval==0:break
        
        return " : ".join([f"{scales[i]:.0f} {units[i]}" for i in reversed(valid)])
    def __repr__(self):
        return f"{self.__class__.__name__}(born={self.s}, pause={self.begin})"
            

class Solution:
    def __init__(self,
                solu=0.,
                solv=0.):
        self.solu=solu
        self.solv=solv
    
    def __iadd__(self,solution):
        if not isinstance(solution,Solution):return self
        self.solu+=solution.solu
        self.solv+=solution.solv
        return self
    def __add__(self,solution):
        if not isinstance(solution,Solution):return self
        return Solution(self.solu+solution.solu,
                        self.solv+solution.solv
                       )
    def __radd__(self,solution):
        if not isinstance(solution,Solution):return self
        return Solution(self.solu+solution.solu,
                        self.solv+solution.solv
                       )
    def __repr__(self):
        return f"{self.__class__.__name__}({self.solu} / {self.solv})"
    
    @property
    def ratio(self):
        if self.solv==0.:return self.solu
        return self.solu/self.solv


class Accumulator:
    def __init__(self,n=None):
        self.data=defaultdict(Solution)
        self.n=n
    def add(self,args:Union[Dict[str,Dict[str,Solution]],
                            Dict[str,Solution],
                            List[Solution],
                            Solution]):
        
        if isinstance(args,list):
            args=enumerate(args)
        elif isinstance(args,dict):
            if isinstance(next(iter(args.values())),Solution):
                args=args.items()
            else:
                args={k:v for curves in args.values() for k,v in curves.items()}.items()
        else:
            args=((0,args),)

#         assert self.n is None or (len(args)==self.n)
        

        for k,v in args:
            self.data[k]+=v

        if self.n is None:
            self.n=len(self.data)

    def reset(self):
        self.data=defaultdict(Solution)

    def __getitem__(self,k):
        assert k in self.data,f"{str(k)} is not in {str(self.data.keys())}"
        return self.data[k]
    
    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self.data)})"
    
    
def use_svg_display():backend_inline.set_matplotlib_formats('svg')
    
    
def set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:axes.legend(legend)
    axes.grid()


class Animator:
    def __init__(self,
                 xlabel=None,
                 ylabel=None,
                 legend=None,
                 xlim=None,ylim=None,
                 xscale='linear',yscale='linear',
                 fmts=('-','m--','r:','g-.',"m-","r--")
                 ):
        if legend is None:legend = []
        use_svg_display()

        self.config_axes = \
            lambda ax,xlabel=xlabel,ylabel=ylabel,xlim=xlim,ylim=ylim,xscale=xscale,yscale=yscale: \
                set_axes(ax,xlabel,ylabel,xlim,ylim,xscale,yscale,legend)
        
        self.X,self.Y=None,None
        self.fmts=fmts
        self.end=False
    
    def init(self,config):
        num_charts=len(config)
        self.fig,self.axes = plt.subplots(num_charts,1,
                                          figsize=(8,5*num_charts))
        if num_charts==1:
            self.axes=[self.axes]

        self.X={fname:{cname:[] for  cname in curves.keys()} for fname,curves in config.items()}
        self.Y={fname:{cname:[] for  cname in curves.keys()} for fname,curves in config.items()}
        # self.L={fname:{cname:cname for  cname in curves.keys()} for fname,curves in config.items()}
        self.fmts_dict={fname:{cname:None for cname in curves.keys()} for fname,curves in config.items()}
        assert len(config)==len(self.axes),"Please set the correct num of metrics and charts in Trainer.add_config!!"
        self.axes_dict={fname:ax for fname,ax in zip(config.keys(),self.axes)}

        i=0
        for fname,curves in config.items():
            for cname in curves.keys():
                self.fmts_dict[fname][cname]=self.fmts[i]
                i+=1

    def add(self,
            x:int=None,
            y:Dict[str,Solution]=None,
            config:Dict[str,Dict[str,Solution]]=None,
            save_pth=None):
        
        '''
        x   : epoch
        y   : metrics
        config : dir structure
        '''
        if self.X is None:
            self.init(config)
        else:
            for ax in self.axes_dict.values():ax.cla() #clear

        
        if x is not None:
            for fname,curves in config.items():
                for cname in curves.keys():
                    self.X[fname][cname].append(x)
                    self.Y[fname][cname].append(y[cname].ratio)


        for fname,xs in self.X.items():
            for cname,x,y in zip(xs.keys(),xs.values(),self.Y[fname].values()):
                self.axes_dict[fname].plot(x,y,
                                           self.fmts_dict[fname][cname],
                                           label=cname)
            
            self.axes_dict[fname].legend()
        
        plt.legend()

        for fname,ax in self.axes_dict.items():
            if "acc" in fname.lower():
                self.config_axes(ax,ylim=(0,1),ylabel=fname)
            else:
                self.config_axes(ax,ylabel=fname)

        if save_pth is not None:
            plt.savefig(f"{save_pth}/draw_config.png")

    def __repr__(self):
        try:
            lastX={cname:val[-1] \
                for fname,curve in self.X.items() \
                    for cname,val in curve.items()}
            lastY={cname:val[-1] \
                for fname,curve in self.Y.items() \
                    for cname,val in curve.items()}
            Xstr=indent(to_str(lastX))
            Ystr=indent(to_str(lastY))
            return f"{self.__class__.__name__}(\n  {Xstr},\n  {Ystr} \n)"
        except:
            return  f"{self.__class__.__name__}(\n None,\n  None \n)"

def removeModule(module_name):
#     assert os.path.isdir(module_name),"Please input folder path!!!"
    path=f"{os.getcwd()}/model/info/{module_name}"
    if not os.path.exists(path):return
    shutil.rmtree(path)

def indent(obj:str,bits=2):
    space=" "*bits
    return ("\n"+space).join(obj.split("\n"))

def to_str(obj,lb="",rb="",exclude=("sos","eos")):
    '''
    pretty print dict
    '''
    if type(obj) is not dict:
        return str(obj)
    return "{\n"+",\n".join(["  "+lb + str(k) + rb+":  " + indent((to_str(v))) \
                       for k,v in obj.items() \
                        if (v!=None) and (k not in exclude)]) +"\n}"

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__=="__main__":
    pass