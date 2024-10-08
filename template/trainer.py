'''---  import utils  ---'''
if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from template.head  import *
from template.utils import *

'''---      self      ---'''
fadir=lambda pth:os.path.dirname(pth)
spth=lambda pth:os.path.split(pth)
sextpth=lambda pth:os.path.splitext(pth)
cwd=fadir(fadir(__file__))

class TrainerCfg:
    config_path=f"{cwd}/configs/"
    init_config_name='init_config.yaml'
    def __init__(
            self,
            update_interval=300,
            empty_cache=True,
            device:str="cuda:0",
            loss_threshold:float=0,
                #   model_name:
            num_epochs:int=None,
            
            lr:float=None,
            weight_decay:float=None,
            scheduler_step_every_batch=False,
            do_validate=False,
            **kwargs,
    ):
        self.info={
            "num_epochs":num_epochs,
            "lr":lr,
            "weight_decay":weight_decay,
            "loss_threshold":loss_threshold,
            "empty_cache":empty_cache,
            "update_interval":update_interval,
            "device":device,
            "load_device":device if isinstance(device,str) else f"cuda:{device[0]}",
            "multiGPU":False if type(device) in (str,int) else True,
            "training_acc":float("-inf"),
            "scheduler_step_every_batch":scheduler_step_every_batch,
            "do_validate":do_validate,
        }
        for k,v in kwargs.items():
            self.info[k]=v


class Trainer:
    '''
    
    default config :init_config.yaml


    below need overriding:

    def init(self,I):
        # init optimizer,criterion etc.
        pass   
        
    def train_epoch(self,model_name,net,I,*data_batch):
        # init batch_data
        # forward -grad clip
        # optimizer -backward -optimizer
        # must save loss !! :

        with torch.no_grad():
            I.loss=loss.detach()
        pass

    '''
    class Info:
        def __init__(self,**kwargs):
            for k,v in kwargs.items():
                setattr(self,k,v)
        
        def __repr__(self):
            return f"{self.__class__.__name__}(\n" + \
                f"{indent(to_str(self.__dict__,lb='<',rb='>'))}\n)"

    def __init__(self,model:Union[nn.Sequential,Any],
                 dataloader:tud.DataLoader,
                 config:Optional[Union[str,TrainerCfg]]=None):
        self.model=self.net=model
        
        self.dataloader=dataloader

        self.tobedef=self.init_config(config)

        if not isinstance(self.config.device,str):
            self.model=nn.DataParallel(self.model,self.config.device)
            # self.dataloader=nn.DataParallel(self.dataloader,self.config.device)

    def load_state_dict(self,model_name,best=""):
        model_pth=f"{self.config.cwd}/model/info/{model_name}/{model_name}{best}_params.pth"
        if os.path.exists(model_pth):
            self.model.load_state_dict(
                torch.load(model_pth)
            )

    def load_model(self):
        self.model.to(self.config.load_device)
    def load_tensor(self,*args):
        out=[]
        for k in args:
            out.append(k.to(self.config.load_device))
        return out
    
    def init_config(self,config:Optional[Union[str,TrainerCfg]]) -> list:
        tobedef=None
        if type(config) is TrainerCfg:
             self.config=config.info
        else:
            if type(config) is str:
                TrainerCfg.config_path,TrainerCfg.init_config_name=spth(config)
            self.config,tobedef=self.getcfg(f"{TrainerCfg.config_path}/{TrainerCfg.init_config_name}")
        
        self.config=Trainer.Info(**self.config)
        return tobedef
    @staticmethod
    def load_config(model_name:str,dirname=None) -> Info:
        dirname=cwd if dirname is None else dirname
        return pickle.load(open(f"{dirname}/model/info/{model_name}/trainer_config.pkl","rb"))
#     def getcfg(self,pth:str) -> Tuple[dict,list]:
#         with open(pth,"r") as f:
#             cfg=yaml.load(f, Loader=yaml.FullLoader)
#         undefined=[]
#         for k,v in cfg.items():
#             if type(v) is not str:continue
#             if v[0]=="!":
#                 var=None
#                 code_obj=compile("var="+v[1:], '<string>', 'exec')
#                 exec(code_obj)
#                 cfg[k]=var
#             elif v=="self":
#                 undefined+=[k]
#             elif "optimizer" in k:
#                 cfg[k]=getattr(opt,v)
#             elif "criterion" in k:
#                 cfg[k]=getattr(nn,v)
#             elif "scheduler" in k:
#                 cfg[k]=getattr(tol,v)

#             '''add more part using eilf here'''
#         return cfg,undefined
    
    def add_config(fadir,**kwargs):
        def wrapper(init):
            def decorator(self,I):
                I.cwd=fadir
                I.loss_threshold=I.loss_threshold
                for k,v in kwargs.items():
                    setattr(I,k,v)
                return init(self,I)
            return decorator
        return wrapper

    def add_timer(fit):
        def decorator(self,*args,**kwargs):
            I=self.config
            I.timer=Timer()
            I.updateTimeStamp=Timer()
            I.cur_steps=0
            I.all_steps=max(len(self.dataloader.sampler)//self.dataloader.batch_size,1)
            
            ret=fit(self,*args,**kwargs)
            return ret
        return decorator         

    def add_animator(fit):
        def decorator(self,*args,**kwargs):
            I=self.config
            I.animator=Animator(xlabel='epoch',ylabel='loss',
                                xlim=[1,I.num_epochs])
            ret=fit(self,*args,**kwargs)
            I.animator.end=True
            
            return ret

        return decorator


    def add_metric(train):
        def decorator(self,net,data_iter,I):
            I=self.config
            I.metric=Accumulator()
            ret=train(self,net,data_iter,I)
            return ret
        
        return decorator


    def add_train_batch(self,empty_cache=False):
        I=self.config
        
        if I.scheduler_step_every_batch:
            if hasattr(I,"scheduler") and I.scheduler is not None:
                I.scheduler.step()
            else:
                for i in range(1,10):
                    if not hasattr(I,f"scheduler{i}"):break
                    if getattr(I,f"scheduler{i}") is None:break
                    getattr(I,f"scheduler{i}").step()        

        with torch.no_grad():
            I.metric.add(I.draw_config)
        if empty_cache:
            if isinstance(I.device,str):
                torch.cuda.empty_cache()
            else:
                for i in I.device:
                    torch.cuda.set_device(torch.device(f"cuda:{i}"))
                    torch.cuda.empty_cache()
                    
    
        I.cur_steps+=1
        if I.updateTimeStamp.interval>I.update_interval:
            I.cur_steps%=I.all_steps
            json.dump({"Epoch":I.e,
                       "Steps":f"{I.cur_steps}/{I.all_steps}",
                       "Takes":f"{I.timer.format(I.timer.stop())}",
                       "LastUpdate":(datetime.now()+timedelta(minutes=6)).strftime("%Y-%m-%d %H:%M:%S")},
                    open(f"{I.model_dir}/running_state.json","w"),
                    indent=4)
            I.updateTimeStamp.restart()

                
    
    def add_train(self):
        I=self.config
        ''' step scheduler '''
        if not I.scheduler_step_every_batch:
            if hasattr(I,"scheduler") and I.scheduler is not None:
                I.scheduler.step()
            else:
                for i in range(1,10):
                    if not hasattr(I,f"scheduler{i}"):break
                    if getattr(I,f"scheduler{i}") is None:break
                    getattr(I,f"scheduler{i}").step()
        
        takes=I.timer.format(I.timer.stop())
        # print("Takes\t",": ", takes)
        
        save_config = collections.defaultdict(dict)
        save_config["Takes"]=takes
        save_config["Epochs"]=I.e
        for fname,curves in I.draw_config.items():
            for cname,curve in curves.items():
                save_config[fname][cname]=I.metric[cname].ratio
        json.dump(save_config,
                  open(f"{I.model_dir}/current_state.json","w"),
                  indent=4)
        
        
        '''step animator'''
        I.animator.add(I.e,
                       I.metric.data,
                       I.draw_config,
                       I.model_dir)

        ''' accumulator '''
        if isinstance(I.loss_threshold,list):
            for m,t in zip(I.metric.data.values(),I.loss_threshold):
                if m.ratio>=t:break
            else:I.stop=True
        elif isinstance(I.loss_threshold,dict):
            for k in I.loss_threshold:
                if I.metric[k].ratio>=I.loss_threshold[k]:break
            else:I.stop=True
        else:
            for m in I.metric.data.values():
                if m.ratio>=I.loss_threshold:break
            else:I.stop=True

    
    def print_result(self,I):
        return
        raise NotImplementedError
        print(f'loss {I.metric[0]/I.metric[1]:.8f},{I.metric[1]/I.timer.stop():.1f}'
            f'tokens/sec on {str(I.device)}')  
        

    def add_fit(fit):
        def decorator(self,*args,**kwargs):
            I=self.config
            if not I.empty_cache:
                self.load_model()
            ret=fit(self,*args,**kwargs)
            self.print_result(I)
            print(f"Training {I.timer.format(I.timer.stop())} .")
            return ret
        
        return decorator
    
    def model_save_condition(self,I):
        '''
        after per whole train epoch

        please update I.training_acc before return
        '''
        return True
    
    def no_grad(func):
        def decorator(self,*args,**kwargs):
            with torch.no_grad():
                return func(self,*args,**kwargs)
        return decorator

    def add_validate_batch(self,I):
        I.validate_cur_steps+=1
        if I.validateTimeStamp.interval>I.update_interval:
            json.dump({"Epoch":I.e,
                       "Steps":f"{I.validate_cur_steps}/{I.validate_all_steps}",
                       "Takes":f"{I.timer.format(I.timer.stop())}",
                       "LastUpdate":(datetime.now()+timedelta(minutes=6)).strftime("%Y-%m-%d %H:%M:%S")},
                    open(f"{I.model_dir}/running_validate.json","w"),
                    indent=4)
            I.validateTimeStamp.restart()

    def add_validate(self,I):
        save_config = collections.defaultdict(dict)
        save_config["Epochs"]=I.e
        save_config["Validate_acc"]=I.validate_acc
        save_config["Validate_acc_list"]=I.validate_acc_list
        json.dump(save_config,
                  open(f"{I.model_dir}/current_validate.json","w"),
                  indent=4)

    @no_grad
    def validate(self,I):
        if not I.do_validate:return True
        model=self.model
        data_iter=I.validate_loader
        I.validate_metric=Accumulator()
        I.validateTimeStamp=Timer()
        I.validate_cur_steps=0
        I.validate_all_steps=max(len(data_iter.sampler)//data_iter.batch_size,1)
        for data_batch in data_iter:
            batch_correct,batch_total=self.validate_batch(model, I, data_batch)
            I.validate_metric.add(Solution(batch_correct,batch_total))
            self.add_validate_batch(I)
        
        I.validate_acc_list.append(I.validate_metric[0].ratio)
        ret= I.validate_acc_list[-1] >= I.validate_acc 
        I.validate_acc=max(I.validate_acc,I.validate_acc_list[-1])
        self.add_validate(I)
        return ret
    
    def validate_batch(self,model,I,data_batch):
        '''
        return tuple(batch_acc,batch_tot)
        '''
        raise NotImplementedError

    def save_model(self,wait,max_interval=180):
                
        I=self.config

        if not self.model_save_condition(I):return
        
        self.save_model_as(I,wait,max_interval,I.model_name)
        
        if not self.validate(I):return
        
        self.save_model_as(I,wait,max_interval,I.model_name+"_best")
        
        if I.timer.interval>max_interval:
            I.timer.restart()
        
    def save_model_as(self,I,wait,max_interval,model_name):

        module=self.model.module if isinstance(self.model,nn.DataParallel) else self.model
        ''' time keeper'''
        if not wait:
            torch.save(module.state_dict(),f"{I.model_dir}/{model_name}_params.pth")
#             pickle.dump(I.config,open(f"{I.model_dir}/trainer_config.pkl","wb"))
            
        elif I.timer.interval>max_interval:
            torch.save(module.state_dict(),f"{I.model_dir}/{model_name}_params.pth")
#             pickle.dump(I.config,open(f"{I.model_dir}/trainer_config.pkl","wb"))
            

    def save_config(self):
        I=self.config
        I.animator.config_axes=None
        with open(f"{I.model_dir}/trainer_config.pkl","wb") as f:
            pickle.dump(I,f)
    
    '''   if not using animator,please delete @init_animator '''
    
    @add_fit
    @add_animator
    @add_timer
    def fit(self,model_name,data_iter=None,validate_loader=None,pre_load:Union[True,None,str]=None,pre_load_best=True):
        if data_iter is None:
            data_iter=self.dataloader
        self.config.dataloader=data_iter
        self.config.batch_size=data_iter.batch_size
        
        I=self.config
        I.stop=False

        self.init(self.config)

        if self.config.do_validate:
            self.config.validate_loader=validate_loader
            self.config.validate_acc=float("-inf")
            self.config.validate_acc_list=[]


        I.model_name=model_name
        I.model_dir=f"{I.cwd}/model/info/{I.model_name}/"
        if not os.path.exists(I.model_dir):
            os.makedirs(I.model_dir)
        
        if pre_load:
            if pre_load==True:
                pre_load=model_name
            best="_best" if pre_load_best else "" 
            self.load_state_dict(pre_load,best)
            self.config.LAST=self.load_config(pre_load)
            if self.config.do_validate and self.config.LAST.do_validate:
                self.config.validate_acc=self.config.LAST.validate_acc
                self.config.validate_acc_list+=self.config.LAST.validate_acc_list
        
        self.model.train()
        
        if not I.empty_cache:
            self.load_model()

        try:
            for e in range(I.num_epochs):
                I.e=e+1
                if I.stop:break
                if I.empty_cache:
                    self.load_model()
                
                self.train(self.model,data_iter,I)
                self.save_model(wait=False)
                
                self.add_train()        
                
                if I.empty_cache:
                    torch.cuda.empty_cache()

        except Exception as e:
            with open(f"{cwd}/model/info/{model_name}/ERROR_while_training.txt","a") as f:
                f.write(type(e).__name__+"\n"+traceback.format_exc()+"\n\n\n")
            raise e
        finally:
            self.save_config()

        return self.model.module \
            if isinstance(self.model,nn.DataParallel) \
            else self.model

    @add_metric
    def train(self,model,data_iter, I):
        for data_batch in data_iter:
            self.train_batch(model, I, data_batch)
            self.add_train_batch()

    '''
    need overriding
    '''
    @add_config(None)
    def init(self,I):
        raise NotImplementedError

    def train_batch(self,model_name,net,I,data_batch):
        raise NotImplementedError
