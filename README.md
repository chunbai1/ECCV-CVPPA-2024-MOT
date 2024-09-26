## 3rd of CVPPA@ECCV'2024: Detection and Multi-Object Tracking of Sweet Peppers Challenge

[challenge link](https://codalab.lisn.upsaclay.fr/competitions/19278#participate)

[official workshop](https://cvppa2024.github.io/challenges/#multi-object-tracking-of-sweet-peppers-challenge)

## create environment

```bash 
## 克隆仓库 clone repository
git clone git@github.com:chunbai1/ECCV-CVPPA-2024-MOT.git
cd ECCV-CVPPA-2024-MOT

## torch
conda create -n bytetrack python=3.7
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

## other package 
pip3 install -r requirements.txt 
python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
```

## Dataset
First, download source dataset from [official Link](https://codalab.lisn.upsaclay.fr/competitions/19278#participate-get_data). Then follow the steps below to process the data into the [MOT20 dataset format](https://blog.csdn.net/OpenDataLab/article/details/125408125).

After downloading the official dataset and decompressing it, the format is as follows:
```bash
.MOT_ECCV_2024
  ├── depth
  ├── mask2former_output
  ├── rgb
  └── weak_labels
```

According to the official dataset description file, the following dataset format is obtained:
```bash
.
├── test
│   ├── depth
│   ├── mask2former_output
│   ├── mask2former_txt
│   ├── rgb
│   └── weak_labels
├── train
│   ├── depth
│   ├── mask2former_output
│   ├── rgb
│   └── weak_labels
└── valid
    ├── depth
    ├── mask2former_output
    ├── rgb
    └── weak_labels
```

Then set the `--subset` parameters to `'train', 'val', 'test'` respectively, and execute the `my_toos/convert2mot20.py` file, convert the `MOT20 data format` to `COCO format` by `my_tools/convert2coco.py`. So the data format is converted to:
```bash
.
├── annotations
│   ├── test.json
│   ├── train.json
│   ├── train_half.json
│   └── val_half.json
├── test
├── train
```

Finally, create a `dataset` directory and establish a `soft link` according to the `my_tools/mix_data.py` file to get the final dataset.

## Train

Change the `exps/example/mot/yolox_x_mix_det.py` configuration to get `exps/example/mot/yolox_x_mix_det_cvppa.py` file. You can refer `ByteTrack_README.md`.

```bash 
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train.py -f exps/example/mot/yolox_x_mix_det_cvppa.py -d 4 -b 16 --fp16 -o \
-c /data/ChaiJM/Competition/CVPPA-DMOT/Code/ByteTrack-main/YOLOX_outputs/yolox_x_mix_det_cvppa/latest_ckpt.pth.tar \
-expn cvppa_resume  -e 4
```

## Test

```bash
CUDA_VISIBLE_DEVICES=7 python tools/demo_track_cvppa.py --save_result -expn cvppa-test5 \
-c pth
```

## Post-Processing
Refer [blog]().