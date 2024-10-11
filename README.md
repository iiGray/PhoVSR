# PhoVSR

## Update

We have released our evaluating recipe for PhoVSR on LRS2, CMLR dataset.

Algorithm 1 is implemented in 'datas/utils', which is only used in data 
preprocessing before training, and not in evaluation.

## Preparation
1. Clone the repository:
```
git clone https://github.com/iiGray/PhoVSR.git
cd PhoVSR
```


2. Set up the environment by:
```
conda create -n phovsr python==3.8
conda activate phovsr
```

## Installation

1. Install torch, torchvision
Download the wheels and install through
[https://download.pytorch.org/whl/torch/](https://download.pytorch.org/whl/torch/)

Version:

**torch : torch-1.13.0+cu117-cp39**

**torchvision: torchvision-0.14.1+cu117-cp39**


2. Install hydra, einops, pypinyin

```
pip install -r requirements.txt
```


## Evaluation

Download models at [PhoVSR models](#PhoVSR-models) and put models into the corresponding folders first.

Print result, the default model is the Chinese lip reading trained from CMLR:
```
python test.py mode=show
```

If memory is low, run like:
```
python test.py mode=show batch_size=2 
```

Change Model:
```
python test.py model_name=LRS2
```
If you want to run the full test set and get the evluation matrics, please download the [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) and [CMLR](https://www.vipazoo.cn/CMLR.html) dataset and change the data at ./data/PhoVSR/datas/info, and preprocess the datas with method used in [VSRML](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/tree/master)  then run.
```
python test.py mode=eval
```
It will calculate the WER/CER. The current default CER for provided 8 samples in this repo is 4.2%.
## PhoVSR models


Put models into the corresponding folders under the folder "model/info"


<details open>

<summary>Lip Reading Sentences 2 (LRS2)</summary>

<p> </p>

|     Components        |                                  url                                         |  size (MB)  |
|:----------------------|-----------------------------------------------------------------------------------:|:-----------:|
|   **VSR Model**       | [GoogleDrive](https://drive.google.com/file/d/1USsLFKtI0xspWqyDHxX2RX0rC7iB9onl/view?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/1YgCUp0rOxVdA6Ww5DvEK6A?pwd=bz4p)   |     202     |
|   **Language Model**  |  [GoogleDrive](https://drive.google.com/file/d/1lEdyGB0JBMkhSpKVOQklA63c_loqtGTH/view?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/1x3LmxkFpxgfMxmnLXKlXfQ?pwd=epmu)   |     196     |

</details>

This VSR model is trained on the 28h LRS2 training set and get 56.2% WER. The language model is trained from plain text datas from LRS2 training set. 


<details open>

<summary>Chinese Mandarin Lip Reading (CMLR)</summary>

<p> </p>

|     Components        |                                 url                                         |  size (MB)  |
|:----------------------|-----------------------------------------------------------------------------------:|:-----------:|
|   **VSR Model**       |  [GoogleDrive](https://drive.google.com/file/d/1g6Oyjl6SjkVwLDYv4BVT2ShQZ_rlsyyu/view?usp=drive_link) or [BaiduDrive](https://pan.baidu.com/s/14IoqyjXF1mFGA5jAiw6ygQ?pwd=exd5)    |     207    |
|   **Language Model**  |  [GoogleDrive](https://drive.google.com/file/d/1VxJlTzb54KZVsY6g7Ra3xLEMvIRs0nFc/view?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/12Ed2Who3CXDAWKJBwfM6Mg?pwd=39vj) |     201    |

</details>

This VSR model is trained on the 61h CMLR training set and get 7.9% CER. The language model is trained from plain text datas from CMLR training set. 

