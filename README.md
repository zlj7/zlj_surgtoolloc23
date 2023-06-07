原仓库链接：https://github.com/bubbliiiing/faster-rcnn-pytorch

在这个基础上修改

## 环境配置

首先`pip install -r requirements.txt`

原始仓库requirements.txt里给的torch和torchvision版本是`torch==1.2.0 torchvision==0.4.0`，和服务器的CUDA版本不太匹配，直接下requirements.txt里的运行会报错`RuntimeError: CUDA error: invalid decice function` 。Google后发现是torch和CUDA版本不匹配导致的，服务器`nvcc -V`显示的CUDA版本是11.6（nvidia-smi显示的是11.7，csdn上说以nvcc -V为准），就照着[这篇CSDN](![img](file:///C:\Users\zlj\AppData\Roaming\Tencent\QQTempSys\%W@GJ$ACOF(TYDYECOKVDYB.png)https://blog.csdn.net/ericdiii/article/details/125258580)安装了对应版本的torch和torchvision（顺手一起装了torchaudio ），即运行

`pip install torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116`



或者直接 `source activate `然后  `conda activate zhenglujie`，这个项目可以不用动直接跑

## 运行

VOC数据集在`/data2/zhenglujie/data/VOCdevkit/`中

先要运行`python voc_annotation.py`， 生成`2007_train.txt`和`2007_val.txt`（不过在服务器上也可以直接用我生成好的）

然后`python train.py`

0号显卡好像一直都挺忙的，我一般用7号显卡

`CUDA_VISIBLE_DEVICES=7 python train.py`