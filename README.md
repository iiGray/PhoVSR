# PhoVSR

## Update

We have released our evaluating recipe for PhoVSR on LRS2, CMLR dataset.

Our Training recipe is coming soon.

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

Calculate the WER/CER:
```
python test.py mode=eval
```
Print result:
```
python test.py mode=show
```

If memory is not enough, run like:
```
python test.py mode=show batch_size=2 
```

Change Model:
```
python test.py model_name=LRS2
```

## PhoVSR models


Put models into the corresponding folders under the folder "model/info"


<details open>

<summary>Lip Reading Sentences 2 (LRS2)</summary>

<p> </p>

|     Components        |  Hours| WER  |                               url                                         |  size (MB)  |
|:----------------------|:-----:|:-----:|-----------------------------------------------------------------------------------:|:-----------:|
|   **VSR Model**       | 28h |    56.2 |[GoogleDrive]() or [BaiduDrive](https://pan.baidu.com/s/1YgCUp0rOxVdA6Ww5DvEK6A?pwd=bz4p)   |     202     |
|   **Language Model**  |     |         | [GoogleDrive]() or [BaiduDrive](https://pan.baidu.com/s/1x3LmxkFpxgfMxmnLXKlXfQ?pwd=epmu)   |     196     |

</details>




<details open>

<summary>Chinese Mandarin Lip Reading (CMLR)</summary>

<p> </p>

|     Components        |  Hours| CER  |                               url                                         |  size (MB)  |
|:----------------------|:-----:|:-----:|-----------------------------------------------------------------------------------:|:-----------:|
|   **VSR Model**       | 61h |    7.9 |    [GoogleDrive](https://drive.google.com/file/d/1g6Oyjl6SjkVwLDYv4BVT2ShQZ_rlsyyu/view?usp=drive_link) or [BaiduDrive](https://pan.baidu.com/s/14IoqyjXF1mFGA5jAiw6ygQ?pwd=exd5)    |     207    |
|   **Language Model**  |     |         |   [GoogleDrive](https://drive.google.com/file/d/1VxJlTzb54KZVsY6g7Ra3xLEMvIRs0nFc/view?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/12Ed2Who3CXDAWKJBwfM6Mg?pwd=39vj) |     201    |

</details>
