'''---  import sys   ---'''
import os,random
import cv2,json
import torch,torchvision
import torch.utils.data as tud
from torchvision import transforms as tfs
import torch.nn.utils.rnn as rnn_utils

from pypinyin import lazy_pinyin
from typing import Literal

'''---------------------'''
_=os.path.dirname(__file__)
cmlr=_+"/info/CMLR/"
# lrs2=_+"/info/LRS2/"
fadir=lambda x:os.path.dirname(x)

class Sampler(tud.sampler.Sampler):
    def __init__(self,st,ed):
        self.st=st
        self.ed=ed
        self.lst=list(range(st,ed))
    
    def __len__(self):
        return self.ed-self.st
    def __iter__(self):
        random.shuffle(self.lst)
        return iter(self.lst)


class Vocab:
    def __init__(self,char_list):
        self.char_list=char_list
        self.char_dict={char_list[i]:i for i in range(len(char_list))}
    def __len__(self):
        return len(self.char_list)
    def __getitem__(self,w):
        if isinstance(w,int):
            return self.char_list[w]
        if isinstance(w,str) and (len(w)==1 or w=="".join(lazy_pinyin(w))):
            return self.char_dict[w] if w in self.char_dict else self.char_dict["<unk>"]
        return [self.__getitem__(k) for k in w]

class AdaptiveSpeed(torch.nn.Module):
    def __init__(self,max_times=1.1):
        super().__init__()
        self.times=max_times
    def forward(self,x):
        cloned=x.clone()
        bound=round(self.times*cloned.size(0))+1
        needs=torch.randint(cloned.size(0),bound,(1,)).item()
        return torch.index_select(cloned,dim=0,
                                  index=torch.linspace(0,cloned.size(0)-1,needs,dtype=torch.int64))
        
    
class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned
    

class Dataset(tud.Dataset):
    def __init__(self,
                 mode:Literal["train","test","val"]="train",
                 convert_gray=False,
                 text_only=False,
                 augment=False,
                 ):
        super().__init__()
        self.mode=mode
        self.convert_gray=convert_gray

        self.text_only=text_only

        if not text_only:
            augments=tfs.Compose([
                tfs.RandomCrop((88,88)),
                tfs.RandomHorizontalFlip(0.5) 
                ]) if (mode=="train" or augment) else tfs.CenterCrop((88,88))
            
            mask=AdaptiveTimeMask(5,25) \
            if (mode=="train" or augment) else tfs.Compose([])
            
            speed=AdaptiveSpeed()\
            if (mode=="train" or augment) else tfs.Compose([])
             
            self.transforms=tfs.Compose([
                tfs.Lambda(self.prenorm),
                augments,
                tfs.Grayscale(),
                mask,
                tfs.Normalize(0.421, 0.165),
            ])
            
        if "train" in mode and "pretrain" not in mode:
            self.mode="train"
#             if convert_gray:
#                 self.transforms.transforms.insert(0,self.togray)
        def collate(self,batch):
            return
        
    def transpose(self,x):
        return x.transpose(0,1)
    
    def prenorm(self,x):return x.permute(0,3,1,2)/255.
    def togray(self,x):
        return torch.tensor([cv2.cvtColor(f.numpy(),cv2.COLOR_RGB2GRAY).tolist()\
                              for f in x]).unsqueeze(-1) 
    
    def setGray(self,mode=True):
        if not (self.convert_gray ^ mode):return
        self.convert_gray=mode
        if mode:self.transforms.transforms.insert(0,self.togray)
        else:self.transforms.transforms.pop(0)


class CMLRDataset(Dataset):
    def __init__(self,  
            mode:Literal["train","val","test"]="train",
            convert_gray=False,
            augment=False,
        ):
        super().__init__(mode,convert_gray,augment=augment)

        self.video_pth=cmlr+"croped_videos/"

        with open(cmlr+mode+".csv","r") as f:
            files=f.readlines()
        self.files=[k.strip().replace("/","#") for k in files]


        config_pth=os.path.join(fadir(fadir(__file__)),"configs","CMLR_model.json")
        char_list=json.load(open(config_pth,"rb"))[2]["char_list"]
        self.vocab=Vocab(char_list)

    def __len__(self):
        return len(self.files)
    
    def getlen(self,i):
        file=self.files[i]
        line=open(self.text_pth+file+".txt","r",encoding="utf-8").readline().strip()
        return line


    def __getitem__(self,i):
        file=self.files[i]
        line=open(self.text_pth+file+".txt","r",encoding="utf-8").readline().strip()
        if self.text_only:
            return line,self.text_pth,file
        video,_,info=torchvision.io.read_video(self.video_pth+file+".mp4",pts_unit='sec')
        if __name__=="__main__":
            return self.transforms(video),line,file
        return self.transforms(video),line

    def collate(self,batch):
        x,t=zip(*batch)
        xl=torch.tensor([k.shape[0] for k in x])
        tl=torch.tensor([len(txt) for txt in t])

        x=rnn_utils.pad_sequence(x)
        t=[torch.tensor(self.vocab[txt]) for txt in t]
        t=rnn_utils.pad_sequence(t)

        return (x.transpose(0,1),xl),(t.transpose(0,1),tl)


class CMLRPhonemeDataset(CMLRDataset):
    def __init__(self,
                 mode:Literal["train","val","test"]="train",
                 convert_gray=False,
                 text_only=False,
                 augment=False):
        super().__init__(mode,convert_gray,augment=augment)
        self.text_only=text_only
        config_pth=os.path.join(fadir(fadir(__file__)),"configs","cmlr_config.json")
        config=json.load(open(config_pth,"rb"))
        char_list=config["char_list"]
        pinyin_list=config["pinyin_list"]
        self.vocab=Vocab(char_list)
        self.phoneme_vocab=Vocab(pinyin_list)
        self.g2p=torch.tensor(config["g2p"])
        self.p2g=torch.tensor(config["p2g"])
    def __getitem__(self, i):
        file=self.files[i]
        
        line=open(self.text_pth+file+".txt","r",encoding="utf-8").readline().strip()
        
        if self.text_only==True:
            return line
        
        video,_,info=torchvision.io.read_video(self.video_pth+file+".mp4",pts_unit='sec')
        return self.transforms(video),line,lazy_pinyin(line)
    
    def collate(self,batch):
        if self.text_only:
            t=batch
            
            tl=torch.tensor([len(txt) for txt in t])
            
            t=[self.vocab[txt] for txt in t]

            xt=[torch.tensor([self.vocab["<sos>"]]+k) for k in t]
            xt=rnn_utils.pad_sequence(xt)
            
            yt=[torch.tensor(k+[self.vocab["<eos>"]]) for k in t]
            yt=rnn_utils.pad_sequence(yt)
            
            
            return (xt.transpose(0,1), tl+1),(yt.transpose(0,1), tl+1)
        x,t,p=zip(*batch)
        xl=torch.tensor([k.shape[0] for k in x])
        tl=torch.tensor([len(txt) for txt in t])

        x=rnn_utils.pad_sequence(x)

        t=[torch.tensor(self.vocab[txt]) for txt in t]
        t=rnn_utils.pad_sequence(t)

        p=[torch.tensor(self.phoneme_vocab[py]) for py in p]
        p=rnn_utils.pad_sequence(p)

        return (x.transpose(0,1),xl), (t.transpose(0,1),tl,p.transpose(0,1))

