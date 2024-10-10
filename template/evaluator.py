import os
from typing import Literal
import torch.utils.data as tud

if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())


from template.head  import *
from template.utils import Timer,Accumulator,Solution

fadir=lambda pth:os.path.dirname(pth)
cwd=fadir(fadir(__file__))



class Evaluator:
    def __init__(self,
                 model,
                 model_name:str,
                 load_best=True,
                 device:str="cuda:0",
                 update_interval=300):

        self.model=model
        self.model_name=model_name

        self.load_best=load_best
        self.load_model(model,model_name=model_name,load_best=load_best,device=device,evaluating=True)
        self.update_interval=update_interval
        self.device=device
        self.indicators={}    

    @staticmethod
    def load_model(model,root_path=cwd,model_dir=None,model_name=None,load_best=False,device="cpu",evaluating=True):
        best="_best" if load_best else ""
        model_dir=model_name if model_dir is None else model_dir
        model.to(device)
        if evaluating:
            model.eval()
        model.load_state_dict(
            torch.load(f"{root_path}/model/info/{model_dir}/{model_name}{best}_params.pth")
        )

    def load_tensor(self,x:Union[List,Tuple]):
        if isinstance(x,torch.Tensor):
            return x.to(self.device)
        return (self.load_tensor(k) for k in x)

    def no_grad(func):
        def decorator(self,*args,**kwargs):
            with torch.no_grad():
                return func(self,*args,**kwargs)
        return decorator
    

    @no_grad
    def __call__(self,dataloader:tud.DataLoader,mode:Literal["eval","show","both"]="show",filename="result"):
        '''
        return : cer if mode includes 'eval'  else None
        '''
        try:
            num_samples=len(dataloader.sampler)
            if mode in ("eval","both"):

                timer=Timer()
                cur_steps=1
                all_steps=len(dataloader.sampler)//dataloader.batch_size
                for indicator in self.indicators.values():
                    indicator.reset()

                for raw_data,ref_data in dataloader:
                    raw_data,ref_data=self.load_tensor([raw_data,ref_data])
                    pred=self.predict(*raw_data)
                    self.evaluate(pred,*ref_data)

                    cur_steps+=1
                    if timer.interval>self.update_interval:
                        jsondict={
                        "Steps":f"{cur_steps}/{all_steps}",
                        "Takes":f"{timer.format(timer.stop())}",
                        "LastUpdate":(datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        jsondict.update({name:indicator.value \
                                        for name,indicator in self.indicators.items()})
                        json.dump(
                            jsondict,
                            open(f"{cwd}/model/info/{self.model_name}/running_eval_{filename}.json","w"),
                            indent=4)
                        timer.restart()
                        
                interval=timer.stop()
                output=f"Evaluate: [ {self.model_name} ]\n"
                output+=f"On  {num_samples} samples.  Takes:  {timer.format(interval)}\n\n"
                output+="\n".join([k+" : "+f"{v.value:.12f}" \
                                for k,v in self.indicators.items()])
                
                with open(f"{cwd}/model/info/{self.model_name}/eval_{filename}.txt","a") as f:
                    f.write(output+"\n\n\n")
                
                print(output)
            if mode in ("show","both"):
                self.show(dataloader)
        except Exception as e:
            with open(f"{cwd}/model/info/{self.model_name}/ERROR_while_evaluating.txt","a") as f:
                f.write(type(e).__name__+"\n"+traceback.format_exc()+"\n\n\n")
            raise e
        
        return output if mode in ("eval","both") else None


    def show(self,dataloader):
        print(f"Predict: (  pred / label )  [ {self.model_name} ]\n"+"_"*100)
        for raw_data,val_data in dataloader:
            raw_data=self.load_tensor(raw_data)
            pred=self.predict(*raw_data)
            self.print(pred,*val_data)

            print("_"*100)

    def evaluate(self,pred,*ref_data):
        '''
        pred  : shape (b, *)
        
        return: tuple (nums of correct or error(not rate), nums)
        '''
        raise NotImplementedError
    def predict(self,*raw_data):
        '''
        return : predict result List[int]
        '''
        raise NotImplementedError

    def print(self,pred, *ref_data):

        '''
        show the predict result of the dataloader
        '''
        raise NotImplementedError
