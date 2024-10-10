# -*- coding: utf-8 -*-
"""
    Author: iiGray
    Description: 
    This file mainly implements algorithm 1, which is used in English
    scenrios, and not needed in Chinese scenrios.

"""

from num2words import num2words
from word2number import w2n
import inflect
from breame.spelling import get_american_spelling, get_british_spelling


import nltk
from nltk.corpus import cmudict
from typing import List,Tuple
# nltk.download('cmudict')
import math
from collections import defaultdict

class NumWords:
    '''
    Get the English representation of the numbers
    '''
    def __init__(self):
        self.engine=inflect.engine()

        self.cache={
            "III"   :"three",
            "dr"    :"doctor"
        }

    def num2word(self,w):
        return num2words(w).replace(",","").replace("-"," ")
    
    def num2decade(self,w):
        if w[2:]=="00":
            return self.num2word(w[:2])+" hundred"
        if w[2]=="0":
            return self.num2word(w[:2])+ " o "+self.num2word(w[-1])
        
        return self.num2word(w[:2])+" "+self.num2word(w[2:])

    def decade2num(self,w:str):
        w_lst=w.split()
        last=w2n.word_to_num(w_lst[-1])
        if len(w_lst)==1:
            return str(last)
        if last==100:
            if len(w_lst)==2:
                return str(w2n.word_to_num(w_lst[0]))+"00"
            if len(w_lst)==3: 
                ret=str(w2n.word_to_num(" ".join(w_lst[:2])))+"00"
                if len(ret)==4:
                    return ret
            return str(w2n.word_to_num(w_lst[0]))+str(w2n.word_to_num(w_lst[1]))
        
        if len(w_lst)==3 and w_lst[-2]=="o":
            return str(w2n.word_to_num(w_lst[0])) +"0"+str(w2n.word_to_num(w_lst[-1]))
        
        return str(w2n.word_to_num(w))

    def isnumeric(self,w):
        if w.replace("'","").isalpha():return False
        if w[0].isalpha():return False
        return True
            

    def n2w(self,w):
        w=str(w)
        if len(w)>=3 and (w[-2:] in ("st","nd","rd","th")) and w[:-2].isdigit():
                eng_num=num2words(int(w[:-2]))
                return self.engine.ordinal(eng_num).replace(",","").replace("-"," ")
        i=0
        while i<len(w) and w[i].isdigit():
            i+=1
        d,c=w[:i],w[i:]
        d=int(d)

        if 1000<=d<=3000 and str(d)[1:3]!="00":
            return self.num2decade(str(d)) + " "+ c.lower()
        
        return self.num2word(d)+" "+c.lower()
    

    def w2n(self,w):
        add=""
        if w.split()[-1]=="s":
            w=" ".join(w.split()[:-1])
            add+="s"

        if w[-2:] in ("st","nd","rd","th"):
            lb=w2n.word_to_num(w) if len(w.split())>1 else 1
            if self.n2w(lb)==w:
                return str(lb)+add
            for i in range(lb,lb+100):
                eng_num=self.num2word(i)
                if self.engine.ordinal(eng_num).replace(",","").replace("-"," ")==w:
                    return str(i)+w[-2:]+add
        
        return self.decade2num(w)+add
        
    def __call__(self,w:str):
        w=w.lower()
        if w in self.cache:
            return self.cache[w]
        try:
            if w[0].isalpha() and w[-1].isdigit():
                i=0
                while w[i].isalpha():i+=1
                s,n=w[:i],w[i:]
                return s+" "+self(n)

            if w.replace(" ","").isalpha():
                return self.w2n(w).strip()
            return self.n2w(w).strip()
        except ValueError:
            return w

import os,pickle
# from phonemizer import phonemize # pip install phonemizer 
#The 'phonemize' function above is time-consuming, so we've already 
# processed all the words we need and saved them in the file below. 
phonemize_dict=pickle.load(open("./configs/lrs2_words_dic.pkl","rb"))
def phonemize(word):
    return phonemize_dict[word]
    


class PhoneticAlignment:
    '''
    The Algorithm1 in paper
    '''
    def __init__(self,merge_phoneme=True,padding="+",ignore_stress=True):
        self.padding=padding
        d = cmudict.dict()
        self.merge_phoneme=merge_phoneme
        self.ignore_stress=ignore_stress

        phonemes = ['AA0', 'AA1', 'AA2', 
                    'AE0', 'AE1', 'AE2', 
                    'AH0', 'AH1', 'AH2', 
                    'AO0', 'AO1', 'AO2', 
                    'AW0', 'AW1', 'AW2', 
                    'AY0', 'AY1', 'AY2', 
                    'B', 'CH', 'D', 'DH', 
                    'EH0', 'EH1', 'EH2', 
                    'ER0', 'ER1', 'ER2', 
                    'EY0', 'EY1', 'EY2', 
                    'F', 'G', 'HH', 
                    'IH0', 'IH1', 'IH2', 
                    'IY0', 'IY1', 'IY2', 
                    'JH', 'K', 'L', 
                    'M', 'N', 'NG', 
                    'OW0', 'OW1', 'OW2', 
                    'OY0', 'OY1', 'OY2', 
                    'P', 'R', 'S', 
                    'SH', 'T', 'TH', 
                    'UH0', 'UH1', 'UH2', 
                    'UW', 'UW0', 'UW1', 'UW2', 
                    'V', 'W', 'Y', 'Z', 'ZH']
        
        multi=["AA","AE","AH",
               "AO","AW","AY",
               "EH","ER","EY",
               "IH","IY","OW","OY","UH"]
        
        self.phoneme_to_ipa = {
            'AA': 'ɑ', 'AA0': 'ɑ', 'AA1': 'ɑ', 'AA2': 'ɑ',
            'AE': 'æ', 'AE0': 'æ', 'AE1': 'æ', 'AE2': 'æ',
            'AH': 'ʌ', 'AH0': 'ə', 'AH1': 'ʌ', 'AH2': 'ʌ',
            'AO': 'ɔ', 'AO0': 'ɔ', 'AO1': 'ɔ', 'AO2': 'ɔ',
            'AW': 'aʊ', 'AW0': 'aʊ', 'AW1': 'aʊ', 'AW2': 'aʊ',
            'AY': 'aɪ', 'AY0': 'aɪ', 'AY1': 'aɪ', 'AY2': 'aɪ',
            'B': 'b',
            'CH': 'tʃ',
            'D': 'd',
            'DH': 'ð',
            'EH': 'ɛ', 'EH0': 'ɛ', 'EH1': 'ɛ', 'EH2': 'ɛ',
            'ER': 'ɝ', 'ER0': 'ɚ', 'ER1': 'ɝ', 'ER2': 'ɝ',
            'EY': 'eɪ', 'EY0': 'eɪ', 'EY1': 'eɪ', 'EY2': 'eɪ',
            'F': 'f',
            'G': 'ɡ',
            'HH': 'h',
            'IH': 'ɪ', 'IH0': 'ɪ', 'IH1': 'ɪ', 'IH2': 'ɪ',
            'IY': 'i', 'IY0': 'i', 'IY1': 'i', 'IY2': 'i',
            'JH': 'dʒ',
            'K': 'k',
            'L': 'l',
            'M': 'm',
            'N': 'n',
            'NG': 'ŋ',
            'OW': 'oʊ', 'OW0': 'oʊ', 'OW1': 'oʊ', 'OW2': 'oʊ',
            'OY': 'ɔɪ', 'OY0': 'ɔɪ', 'OY1': 'ɔɪ', 'OY2': 'ɔɪ',
            'P': 'p',
            'R': 'ɹ',
            'S': 's',
            'SH': 'ʃ',
            'T': 't',
            'TH': 'θ',
            'UH': 'ʊ', 'UH0': 'ʊ', 'UH1': 'ʊ', 'UH2': 'ʊ',
            'UW': 'u', 'UW0': 'u', 'UW1': 'u', 'UW2': 'u',
            'V': 'v',
            'W': 'w',
            'Y': 'j',
            'Z': 'z',
            'ZH': 'ʒ'
        }

        self.ipa_to_phoneme = {
            'ɑ': 'AA', 'æ': 'AE', 'ʌ': 'AH',"ɑː":"AH2", 'ə': 'AH0',"ɐ":"AH0","ɚ":"AH1","ɜː":"AH2", 'ɔ': 'AO',"ɔ̃":"AO","ɔː":"AO2", 'aʊ': 'AW',
            'aɪ': 'AY', 'b': 'B', 'tʃ': 'CH', 'd': 'D', 'ð': 'DH', 'ɛ': 'EH',
            'ɝ': 'ER', 'ɚ': 'ER0', 'eɪ': 'EY', 'f': 'F', 'ɡ': 'G', 'h': 'HH',
            'ɪ': 'IH',"ᵻ":"IH", 'i': 'IY',"iː":"IY2", 'dʒ': 'JH', 'k': 'K',"x": "K", 'l': 'L',"ɬ":"L", 'm': 'M',
            'n': 'N',"n̩":"N", 'ŋ': 'NG', 'oʊ': 'OW',"o":'OW', 'ɔɪ': 'OY', 'p': 'P', 'ɹ': 'R',"r":"R","ɾ":"R",
            's': 'S', 'ʃ': 'SH', 't': 'T',"ʔ":"T", 'θ': 'TH', 'ʊ': 'UH', 'u': 'UW',"uː":"UW1",
            'v': 'V', 'w': 'W', 'j': 'Y', 'z': 'Z', 'ʒ': 'ZH',"oː":"AO2",
        }


        self.built=False
        self.pwdic=defaultdict(lambda: defaultdict(int))
        self.wpdic=defaultdict(lambda: defaultdict(int))
        self.cmudict=d

        self.supplement={
            
            "archers":['AA1', 'R', 'CH', 'ER0', "Z"],
            "mashing":['M', 'AE1', 'SH','IH0', 'NG'],
        }
        #some special and non-word letter combinations need setting manually
        self.cache={
            "0"      :(["0"],['Z+IH+R+OW']),
            "1"      :(["1"],['W+AH+N']),
            "2"      :(["2"],['T+UW']),
            "3"      :(["3"],['TH+R+IY']),
            "4"      :(["4"],['F+AO+R']),
            "5"      :(["5"],["F+AY+V"]),
            "6"      :(["6"],['S+IH+K+S']),
            "7"      :(["7"],['S+EH+V+AH+N']),
            "8"      :(["8"],['EY+T']),
            "9"      :(["9"],['N+AY+N']),
            "ii"     :(["ii"],['T+UW']),
            "iii"    :(["iii"],['TH+R+IY']),
            "iv"     :(["iv"],['F+AO+R']),
            "vi"     :(["vi"],['S+IH+K+S']),
            "vii"    :(["vii"],['S+EH+V+AH+N']),
            "viii"   :(["viii"],['EY+T']),
            "dr"     :(["d","r"],["D+AA+K","T+ER"]),
            "y2k"    :(["y","2","k"],["W+AY","T+UW","K+EY"]),
            "gr4s"   :(["g","r","4","s"],['JH+IY',"AA+R",'F+AO+R',"Z"]),
            "gt3s"   :(['g','t','3','s'], ['JH+IY','T+IY','TH+R+IY',"Z"]),
            "ak47s"  :(['a', 'k', '4', '7','s'], ['AE', 'K', 'F+AO+R+T+IY', 'S+EH+V+AH+N','Z']),
            "20kph"  :(['2','0','k','p','h'],['T+W+EH+N', 'T+IY','K+AH+L+AA+M+AH+T+ER+Z','P+ER','AW+ER']),
            "3'9"    :(["3","'","9"],['TH+R+IY','F+UH+T','N+AY+N']),
            "v12s"   :(['v','1', '2','s'], ['V+IY','T+W+EH+L', 'V','Z']),
            "za298"  :(['z', 'a','2','9','8'], ['Z', 'AH','T+UW','N+AY+N','EY+T']),
            "h2o"    :(['h','2','o'], ['EY+CH','T+UW','OW']),
            "7'2"    :(['7',"'",'2'], ['S+EH+V+AH+N','F+UH+T','T+UW']),
            "b52s"   :(['b','5', '2', 's'], ['B+IY','F+IH+F+T+IY', 'T+UW', 'Z'])
            
        }
        print("Adding words...")
        for word in self.cmudict:
            self.add(word)
        
        for v in self.cache.values():
            self.pwdic[v[1][0]][v[0][0]]+=10000000
            self.wpdic[v[0][0]][v[1][0]]+=10000000
            
        for i in range(26):
            c=chr(i+ord("a"))
            p=self.g2p(c)
            p=[k if k[-1].isalpha() else k[:-1] for k in p]
            self.wpdic[c]["+".join(p)]+=10000000
            self.pwdic["+".join(p)][c]+=10000000
            


        self.nw=NumWords()

        # print("Adding nums...")
        for num in range(0,100):
            self.add_num(num,1000)
            self.add_num(str(num)+"s",1000)
            if num%10==1:
                self.add_num(str(num)+"st",1000)
            elif num%10==2:
                self.add_num(str(num)+"nd",1000)
            elif num%10==3:
                self.add_num(str(num)+"rd",1000)
            elif num!=0:
                self.add_num(str(num)+"th",1000)
            if num%100==0:
                print(num,"over!")

        for kw in self.wpdic:
            for kp in self.wpdic[kw]:
                self.wpdic[kw][kp]=math.log(self.wpdic[kw][kp])

        for kp in self.pwdic:
            for kw in self.pwdic[kp]:
                self.pwdic[kp][kw]=math.log(self.pwdic[kp][kw])
        
        print("WordSegment Built Over!!")

    def ipa2p(self,ipa):
        l,r=0,1
        ret=[]
        nums=0
        while r<=len(ipa):
            nums+=1
            if ipa[l:r] not in self.ipa_to_phoneme:
                nums+=1
                if r>l+1:
                    nums+=1
                    ret.append(self.ipa_to_phoneme[ipa[l:r-1]])
                    l=r-1
            if nums>=10000000:
                raise Exception("ipa:"+ ipa)
            r+=1
        if l<len(ipa):
            ret.append(self.ipa_to_phoneme[ipa[l:]])
        return ret

    def num_g2p(self,w):
        p=[]
        try:
            word=self.nw(w).split()
        except Exception as e:
            print(w)
            raise e
        for wd in word:
            p.extend(self.g2p(wd))
        return p
        
    def g2p(self,w):
        if w not in self.cmudict:
            for i in range(10):
                if str(i) in w:
                    return self.num_g2p(w)

        if w in self.supplement:
            return self.supplement[w]
        try:
            return self.cmudict[w][0]
        except:
            try:
                ret=self.cmudict[get_american_spelling(w)][0]
            except:
                tail=[]
                if w[-2:]=="'s":
                    tail=["Z"]
                    w=w[:-2]
                    if w not in self.cmudict:
                        w=get_american_spelling(w)

                    if w in self.cmudict:
                        return self.cmudict[w][0]+tail
                    
                ipa=phonemize(w.lower()).strip()
                if "ɹoʊmən " in ipa: ipa=ipa.split()[-1]

                ret=self.ipa2p(ipa)+tail
                
            return ret


    def merge(self,r:list,i,pref,l,ret:list,inter=""):
        if(i>=len(r)):return
        if l==0:
            ret.append(pref+r[i:])
            return
        if l+1==len(r[i:]):
            ret.append(pref+[inter.join(r[i:])])
            return

        for n_ in range(l+1):
            self.merge(r,i+n_+1,pref+[inter.join(r[i:i+n_+1])],l-n_,ret,inter)

    def add_gp(self,g:list,p:list):
        gs,ps=[],[]
        if len(g)<len(p):
            gs=[g]
            self.merge(p,0,[],len(p)-len(g),ps,inter="+")

        else:
            ps=[p]
            self.merge(g,0,[],len(g)-len(p),gs,inter="")

        for g in gs:
            for p in ps:
                for i in range(len(p)):
                    self.pwdic[p[i]][g[i]]+=1
                    self.wpdic[g[i]][p[i]]+=1

    def add_num(self,num,times=10000):
        w=str(num)
        p=self.num_g2p(w)
        p=[c if c[-1].isalpha() else c[:-1] for c in p]
        w=list(w)
        self.add_gp(w,p)


    def add(self,w):
        w=w.lower()
        assert self.padding not in w, \
            f"Word: {w}  has the padding value '{self.padding}'!"
        p=self.g2p(w)
        p=[k[:-1] if k[-1].isdigit() else k for k in p]
        w=list(w)

        self.add_gp(w,p)


    def split(self,w):
        if w[-2:]=="'s":
            ret=self.split(w[:-2])
            return ret[0]+["'s"],ret[1]+["Z"]
        if w[-1]=="s" and w[:-1].isdigit():
            return list(w),self.split(w[:-1])[1]+["Z"]
        if w in self.cache:
            return self.cache[w]

        p=self.g2p(w)
        p=[k[:-1] if k[-1].isdigit() else k for k in p]

        w=list(w.lower())

        prefs=[]
        if len(p)<=len(w):
            ws=[]
            self.merge(w,0,[],len(w)-len(p),ws,inter="")
            for w_ in ws:
                tw,tp,prob=[],[],0.
                for i in range(len(p)):
                    tw.append(w_[i])
                    tp.append(p[i])
                    prob+=self.wpdic[w_[i]][p[i]]

                prefs.append(((tw,tp),prob))
        else:
            ps=[]
            self.merge(p,0,[],len(p)-len(w),ps,inter="+")
            for p_ in ps:
                tw,tp,prob=[],[],0.
                for i in range(len(w)):
                    tw.append(w[i])
                    tp.append(p_[i])
                    prob+=self.pwdic[p_[i]][w[i]]

                prefs.append(((tw,tp),prob))

        prefs.sort(key=lambda x:-x[-1])

        return prefs[0][0]


if __name__=="__main__":
    '''
    Here's an example how to use it
    '''

    ps=PhoneticAlignment()


    for word in ["hello","world","how","are","you"]:
        grapheme,phoneme=ps.split(word)

        print(word,":\t",grapheme,phoneme)
    

