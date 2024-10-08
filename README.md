# PhoVSR

## Update

We have released our evaluating recipe for PhoVSR on LRS2, CMLR


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

3. Install torch, torchvision
Download the wheels and install through
[https://download.pytorch.org/whl/torch/](https://download.pytorch.org/whl/torch/)

Version:

**torch : torch-1.13.0+cu117-cp39**

**torchvision: torchvision-0.14.1+cu117-cp39**


4. Install hydra, einops, pypinyin

```
pip install -r requirements.txt
```


## Evaluation
Calculate the WER/CER:
```
python test.py mode=eval
```
Print result
```
python test.py mode=show
```

If memory is not enough, run like
```
python test.py mode=show batch_size=2 beam_size=3
```


## PhoVSR models


Put models in the corresponding folders under the folder "model/info"


<details open>

<summary>Lip Reading Sentences 2 (LRS2)</summary>

<p> </p>

|     Components        |  Hours| WER  |                               url                                         |  size (MB)  |
|:----------------------|:-----:|:-----:|-----------------------------------------------------------------------------------:|:-----------:|
|   **VSR Model**       | 28h |    56.2 |[BaiduDrive]()   |     203     |
|   **Language Model**  |     |         |  [BaiduDrive]()   |     196     |

</details>




<details open>

<summary>Chinese Mandarin Lip Reading (CMLR)</summary>

<p> </p>

|     Components        |  Hours| CER  |                               url                                         |  size (MB)  |
|:----------------------|:-----:|:-----:|-----------------------------------------------------------------------------------:|:-----------:|
|   **VSR Model**       | 61h |    7.9 |   [BaiduDrive](https://pan.baidu.com/s/14IoqyjXF1mFGA5jAiw6ygQ?pwd=exd5)     |     207    |
|   **Language Model**  |     |         |  [BaiduDrive](https://pan.baidu.com/s/12Ed2Who3CXDAWKJBwfM6Mg?pwd=39vj)   |     201    |

</details>