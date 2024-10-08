'''---  import sys   ---'''
import sys,time,collections,math,os,random
import cv2,json,pickle
import numpy as np,pandas as pd
from PIL import Image
import torch,torchvision
import torch.utils.data as tud
from torchvision import transforms as tfs
import torch.nn.utils.rnn as rnn_utils
# import pypinyin
from pypinyin import lazy_pinyin
from typing import Literal
from datas.sampler import *

'''---------------------'''
_=os.path.dirname(__file__)
cmlr=_+"/info/CMLR/"
lrs2=_+"/info/LRS2/"
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
                 mode:Literal["train","test","val","all"]="train",
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
                
#                 tfs.Normalize(0.421, 0.165) if self.convert_gray else\
#                 tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            mode:Literal["train","test","all"]="train",
            convert_gray=False,
            augment=False,
        ):
        super().__init__(mode,convert_gray,augment=augment)

        self.video_pth=cmlr+"croped_videos/"
        self.text_pth=cmlr+"texts/"
        if mode=="all":
            files=os.listdir(self.video_pth)
            self.files=[k[:-4] for k in files]
        else:
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
        return line,self.text_pth,file
        video,_,info=torchvision.io.read_video(self.video_pth+file+".mp4",pts_unit='sec')
        # fps=info["video_fps"]
        # raw=open(self.text_pth+file[:-3]+"txt","r",encoding="utf-8").read().splitlines()
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


class PinyinDataset(CMLRDataset):
    def __init__(self,
                 mode:Literal["train","test","all"]="train",
                 convert_gray=False):
        super().__init__(mode,convert_gray)
        config_pth=os.path.join(fadir(fadir(__file__)),"configs","CMLR_model.json")
        pinyin_list=json.load(open(config_pth,"rb"))[2]["pinyin_list"]
        self.phoneme_vocab=Vocab(pinyin_list)
    def __getitem__(self, i):
        file=self.files[i]
        video,_,info=torchvision.io.read_video(self.video_pth+file+".mp4",pts_unit='sec')
        line=open(self.text_pth+file+".txt","r",encoding="utf-8").readline().strip()
        return self.transforms(video),line,lazy_pinyin(line)
    
    def collate(self,batch):
        x,t,p=zip(*batch)
        xl=torch.tensor([k.shape[0] for k in x])
        tl=torch.tensor([len(txt) for txt in t])

        x=rnn_utils.pad_sequence(x)

        t=[torch.tensor(self.vocab[txt]) for txt in t]
        t=rnn_utils.pad_sequence(t)

        p=[torch.tensor(self.phoneme_vocab[py]) for py in p]
        p=rnn_utils.pad_sequence(p)

        return (x.transpose(0,1),xl), (t.transpose(0,1),tl,p.transpose(0,1))

class CMLRPhonemeDataset(CMLRDataset):
    def __init__(self,
                 mode:Literal["train","test","all"]="train",
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
        self.w2p=torch.tensor(config["pam"])
        self.p2w=torch.tensor(config["map"])
        self.c2p=torch.tensor(config["c2p"])
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


class LRS2Dataset(Dataset):
    def __init__(self,
                 mode="train",
                 convert_gray=True,
                 text_only=False,
                augment=False,
                cleaned=True):
        
        super().__init__(mode,convert_gray,text_only,augment=augment)

        self.data_dir=lrs2+"/main/"
        
        smode=mode
        if ("train" in mode) and ("pretrain" not in mode) :smode="train"
        self.train_list=open(lrs2+f"{smode}.txt","r").readlines()
        self.lengths=pickle.load(open(lrs2+"/lrs2_lengths.pkl","rb"))
        
        if mode=="pretrain":
            self.data_dir=lrs2+"/pretrain/"
            self.train_list=pickle.load(open("t_list.pkl","rb"))
            self.lengths=pickle.load(open("l_list.pkl","rb"))
            self.split_idx=55104
        if mode=="train":
            split_idx=pickle.load(open("split_int.pkl","rb"))
            self.train_list=pickle.load(open("ot_list.pkl","rb"))[split_idx:]
            self.lengths=pickle.load(open("ol_list.pkl","rb"))[split_idx:]
            
        
        config_name="LRS2_model.json"
        
        config_pth=os.path.join(fadir(fadir(__file__)),"configs",config_name)

        char_list=json.load(open(config_pth,"rb"))[2]["char_list"]

            
        self.vocab=Vocab(char_list)
        
        if mode=="pretrain" and ((not text_only) or cleaned):
            if os.path.exists("ot_list.pkl"):
                self.train_list=pickle.load(open("ot_list.pkl","rb"))
                self.lengths=pickle.load(open("ol_list.pkl","rb"))
                self.split_idx=pickle.load(open("split_int.pkl","rb"))
            else:
                
                train_list=[]
                lengths=[]
                for i in range(55104):
                    if i%1000==0:print(i)
                    path=self.train_list[i].strip()
                    line=list(open(lrs2+"/pretrain/"+path+".txt","r",encoding="utf-8").readline()[5:].strip())
                    line=[k if k!=" " else "<sep>" for k in line]
                    t=torch.tensor(self.vocab[line])
                    needs=1+t.size(0)+(t[1:]==t[:-1]).sum().item()
                    if self.lengths[i]>=needs:
                        lengths.append(self.lengths[i])
                        train_list.append(path)
                self.split_idx=len(train_list)
                
                for i in range(55104,len(self.train_list)):
                    if i%1000==0:print(i)
                    path=self.train_list[i].strip()
                    line=list(open(lrs2+"/main/"+path+".txt","r",encoding="utf-8").readline()[5:].strip())
                    line=[k if k!=" " else "<sep>" for k in line]
                    t=torch.tensor(self.vocab[line])
                    needs=1+t.size(0)+(t[1:]==t[:-1]).sum().item()
                    if self.lengths[i]>=needs:
                        lengths.append(self.lengths[i])
                        train_list.append(path)
                        
                self.lengths=lengths
                self.train_list=train_list
                
                pickle.dump(self.train_list,open("ot_list.pkl","wb"))
                pickle.dump(self.lengths,open("ol_list.pkl","wb"))
                pickle.dump(self.split_idx,open("split_int.pkl","wb"))
        
        
    def __len__(self):
        return len(self.train_list)
    
    def __getitem__(self, i):
        if self.mode=="pretrain" and i>=self.split_idx:
            self.data_dir=lrs2+"/main/"
        elif self.mode=="pretrain":            
            self.data_dir=lrs2+"/pretrain/"
            
        path=self.train_list[i].strip()
        if self.mode=="test":path=path.split()[0]
            
        line=list(open(self.data_dir+path+".txt","r",encoding="utf-8").readline()[5:].strip())
        line=[k if k!=" " else "<sep>" for k in line]
        if self.text_only:
#             return path
            return line,path
        
            
        video,_,info=torchvision.io.read_video(self.data_dir+path+".mp4",pts_unit='sec')
#         line=open(self.data_dir+path+".txt","r",encoding="utf-8").readline()[5:].strip()
        
        return self.transforms(video),line
        
    def collate(self,batch):
        if self.text_only:
            t,*args=zip(*batch)
            tl=torch.tensor([len(txt) for txt in t])
            
            
            t=[self.vocab[txt] for txt in t]
            
            xt=[torch.tensor([self.vocab["<sos>"]]+k) for k in t]
            xt=rnn_utils.pad_sequence(xt)
            
            yt=[torch.tensor(k+[self.vocab["<eos>"]]) for k in t]
            yt=rnn_utils.pad_sequence(yt)
            
            
            return (xt.transpose(0,1), tl+1),(yt.transpose(0,1), tl+1)
            
            
            
        x,t=zip(*batch)
        xl=torch.tensor([k.shape[0] for k in x])
        tl=torch.tensor([len(txt) for txt in t])

        x=rnn_utils.pad_sequence(x)

        t=[torch.tensor(self.vocab[txt]) for txt in t]
        t=rnn_utils.pad_sequence(t)



        return (x.transpose(0,1),xl), (t.transpose(0,1),tl)
    
    
class LRS2PhonemeDataset(Dataset):
    def __init__(self,
                 mode="train",
                 convert_gray=True,
                 text_only=False,
                 use_full_config=False,
                 augment=False,
                 merge_sep=False,
                 no_sep_phoneme=False,
                 pretrain_vocab=False):
        
        super().__init__(mode,convert_gray,text_only,augment)
        self.merge_sep=merge_sep
        self.data_dir=lrs2+"/main/"
        self.label_dir=lrs2+"/main_phonetic/"
        if mode=="pretrain":
            self.data_dir=lrs2+"/pretrain_clipped/"
            self.label_dir=lrs2+"/pretrain_phonetic/"
        if self.mode=="all":
            self.train_list=open(lrs2+"train.txt","r").readlines()+\
            open(lrs2+"val.txt","r").readlines()
            tsts=open(lrs2+"test.txt","r").readlines()
            self.train_list+=[k.strip().split()[0] for k in tsts]
        else:
            self.train_list=open(lrs2+f"{self.mode}.txt","r").readlines()
        self.lengths=pickle.load(open(lrs2+"/lrs2_lengths.pkl","rb"))
        if mode=="pretrain":
            if os.path.exists("tt_list.pkl") and os.path.exists("ll_list.pkl"):
#                 self.train_list=pickle.load(open("tt_list.pkl","rb"))
#                 self.lengths=pickle.load(open("ll_list.pkl","rb"))
#                 self.split_idx=96280#pickle.load(open("split_idx.pkl","rb"))
                self.train_list=pickle.load(open("list_lt_1800.pkl","rb"))
                self.lengths=pickle.load(open("lengths_lt_1800.pkl","rb"))
                self.split_idx=96280-14#pickle.load(open("split_idx.pkl","rb"))
            else:
                current_list=[]
                lengths=pickle.load(open(lrs2+"/lrs2_pretrain_lengths.pkl","rb"))
                notinfs=pickle.load(open(lrs2+"/lrs2_notinfs.pkl","rb"))
                self.lengths=[]
                for i,path in enumerate(self.train_list):
                    if i%1000==0:print(i)
                    if os.path.exists(self.data_dir+path.strip()+"_clipped.mp4") \
                    and os.path.exists(self.label_dir+path.strip()+".txt") and lengths[i]<=3000  and notinfs[i]:
                        current_list.append(path)
                        self.lengths.append(lengths[i])
                
                self.train_list=current_list
                
                self.split_idx=len(self.train_list)
                pickle.dump(self.split_idx,open("split_idx.pkl","wb"))
                self.train_list+=open(lrs2+f"train.txt","r").readlines()
                self.lengths+=pickle.load(open(lrs2+"/lrs2_lengths.pkl","rb"))
                pickle.dump(self.train_list,open("t_list.pkl","wb"))
                pickle.dump(self.lengths,open("l_list.pkl","wb"))
        config_name="lrs2_full_config.json" if use_full_config else "lrs2_config.json"

        self.no_sep_phoneme=no_sep_phoneme
        if merge_sep:
            config_name="lrs2_merge_sep_config.json"
            if no_sep_phoneme:
                config_name="lrs2_merge_config.json"
        if mode=="pretrain":
            config_name="lrs2_pretrain_config.json"
        if pretrain_vocab:
            config_name="lrs2_pretrain_config.json"
        config_pth=os.path.join(fadir(fadir(__file__)),"configs",config_name)
        config=json.load(open(config_pth,"rb"))
        char_list=config["char_list"]
        phoneme_list=config["phoneme_list"]

        self.w2p=torch.tensor(config["w2p"])
        self.c2p=config["c2p"]
        self.p2w=torch.tensor(config["p2w"])

        self.vocab=Vocab(char_list)
        self.phoneme_vocab=Vocab(phoneme_list)
        self.mode=mode
    def __len__(self):
        return len(self.train_list)
    
    def __getitem__(self, i):
        if self.mode=="pretrain" and i>=self.split_idx:
            self.data_dir=lrs2+"/main/"
            self.label_dir=lrs2+"/main_phonetic/"
        elif self.mode=="pretrain":            
            self.data_dir=lrs2+"/pretrain_clipped/"
            self.label_dir=lrs2+"/pretrain_phonetic/"
            
        path=self.train_list[i].strip()
        if self.mode=="test":
            path=path.split()[0]
#         if self.mode=="test":path=path.split()[0]
#         if self.text_only:
#             return path
#             return open(self.data_dir+path+".txt","r",encoding="utf-8").readline()[5:].strip(),path
        with open(self.label_dir+path+".txt","r",encoding="utf-8") as f:
            line=f.readline().strip().split()
    #             if not self.text_only:
            phoneme=f.readline().strip().split()
            if self.merge_sep:
                line=line+["<sep>"]
                line=(" ".join(line)).split("<sep>")
                line.pop()
                line=[w for k in line for w in (k.strip()+"<sep>").split()]
                if self.no_sep_phoneme:
                    phoneme=(" ".join(phoneme)).replace("<sep>","").strip().split()
                else:
                    phoneme=phoneme+["<sep>"]
                    phoneme=(" ".join(phoneme)).split("<sep>")
                    phoneme.pop()
                    phoneme=[w for k in phoneme for w in (k.strip()+"<sep>").split()]
               

            
            
        if self.text_only:
            return line,phoneme,path
        video,_,info=torchvision.io.read_video(self.data_dir+path+"_clipped.mp4",pts_unit='sec')

        return self.transforms(video),line,phoneme
        
    def collate(self,batch):
        if self.text_only:
            t,*args=zip(*batch)
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

from datas.bucket import *
import cv2
class LD(LRS2PhonemeDataset):
    def __getitem__(self,i):
        path=self.train_list[i].strip()
        cap=cv2.VideoCapture(self.data_dir+path+"_clipped.mp4")
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
        return torchvision.io.read_video(self.data_dir+path+"_clipped.mp4",pts_unit='sec')[0]

class LRS2BucketDataset(tud.Dataset):
    def __init__(self,
                 mode="pretrain",
                 convert_gray=True,
                 text_only=False,
                 augment=True,
                 merge_sep=True,
                 no_sep_phoneme=True,
                 max_frames=1800,
                 num_buckets=50,
                 shuffle=True):
        self.dataset=LRS2PhonemeDataset(mode="pretrain",
                                        convert_gray=convert_gray,
                                        text_only=text_only,
                                        augment=augment,
                                        merge_sep=merge_sep,
                                        no_sep_phoneme=no_sep_phoneme)
        self.vocab=self.dataset.vocab
        lengths=[]
        
#         for i in range(len(self.dataset)):
#             if i%1000==0:print(i)
#             lengths.append(self.dataset[i][0].shape[0])
        lengths=self.dataset.lengths
        
#         self.bucketdataset=CustomBucketDataset(self.dataset,
#                                                lengths,
#                                                max_frames,
#                                                num_buckets,
#                                                shuffle=shuffle)
#         self.batch_sampler=CustomBatchSampler(self.bucketdataset.batches)
        self.batch_sampler=CustomBatchSampler(lengths,max_frames,num_buckets,shuffle)
#         self.batch_sampler=BatchSampler(lengths,max_frames,num_buckets,shuffle)
    def __getitem__(self,idx):
        return self.dataset[idx]
    
        batch=self.bucketdataset[idx]
        return self.dataset.collate(batch)
            
    def __len__(self):
        return len(self.dataset)
        return len(self.bucketdataset)

class MixedDataset(Dataset):
    def __init__(self,
                 mode="train",
                 convert_gray=True):
        super().__init__(mode,convert_gray)

        config_pth=os.path.join(fadir(fadir(__file__)),"configs","MIXED_model.json")

        config=json.load(open(config_pth,"rb"))

        lrs2_char_list=config["LRS2_char_list"]
        cmlr_char_list=config["CMLR_char_list"]
        phoneme_char_list=config["phoneme_char_list"]
        self.pvocab=Vocab(phoneme_char_list)

        self.vocabs=[Vocab(lrs2_char_list),Vocab(cmlr_char_list)]


        self.lrs2_dir=lrs2+"/main/"
        self.cmlr_dir=cmlr
        
        self.lrs2_file_list=open(lrs2+f"{mode}.txt","r").readlines()
        with open(cmlr+mode+".csv","r") as f:
            files=f.readlines()
        self.cmlr_file_list=[k.strip().replace("/","#") for k in files]

        self.lengths=[len(self.lrs2_file_list),len(self.cmlr_file_list)]
                     
    def __len__(self):
        return sum(self.lengths)
    def __getitem__(self,k):
        i=0
        for t,l in enumerate(self.lengths):
            if k>=l:
                k-=l
                i=t+1
            else:
                break

        if i==0:
            path=self.lrs2_file_list[k].strip()
            video_path=self.lrs2_dir+path+"_clipped.mp4"
            text_path=self.lrs2_dir+path+"_phonetic.txt"
        elif i==1:
            path=self.cmlr_file_list[k]
            video_path=self.cmlr_dir+"/croped_videos/"+path+".mp4"
            text_path=self.cmlr_dir+"/phonetic_texts/"+path+".txt"
        
        video,_,info=torchvision.io.read_video(video_path,pts_unit="sec")
        with open(text_path,"r",encoding="utf-8") as f:
            p=f.readline().strip().split()
            t=f.readline().strip().split()

        return self.transforms(video),p,t,i
    
    def collate(self,batch):
        batch.sort(key=lambda x:x[-1])
        
        x,p,y,t=zip(*batch)
        bound=(0,)+t
        xl=torch.tensor([k.shape[0] for k in x])
        pl=torch.tensor([len(phoneme) for phoneme in p])
        yl=torch.tensor([len(txt) for txt in y])

        x=rnn_utils.pad_sequence(x)

        p=[torch.tensor(self.pvocab[phoneme]) for phoneme in p]
        p=rnn_utils.pad_sequence(p)

        y=[torch.tensor(self.vocabs[i][txt]) for i,txt in zip(t,y)]
        y=rnn_utils.pad_sequence(y)

        t=torch.tensor(t) #(batch_size,)

        bounds=torch.tensor(bound).cumsum(dim=0) #(batch_size+1,)
        
        return (x.transpose(0,1),xl),(p.transpose(0,1),pl,y.transpose(0,1),yl,t,bounds)




if __name__=="__main__":

    dataset=Dataset(mode="train",convert_gray=True)
    l=0
    for i in range(len(dataset)):
        if(i%1000==0):print(i)
        l=max(l,len(dataset.getlen(i)))
    # print(dataset.getlen(0))


    exit(0)
    dataset=PinyinDataset(mode="train",convert_gray=True)
    loader=tud.DataLoader(dataset,batch_size=3,collate_fn=dataset.collate)
    sdataset=SyllableDataset(mode="train",convert_gray=True)
    sloader=tud.DataLoader(sdataset,batch_size=3,collate_fn=dataset.collate)
    
    for (x,xl),(y,yl) in sloader:
        print(x.shape,y.shape,yl)
        break
