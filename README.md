## [Recusive Contour Saliency Blending Network for Accurate Salient Object Detection](https://www.google.com)[Under Review]
This repo is the original RCSB playground, with full log of history and codes.
The one RCSB-PyTorch is for review and publication purpose, and it is a simplified version (some logging functions are trimmed)

### Network Architecture

### Prerequisites
Python==3.8.3\
Torch==1.8.0+cu111\
Torchvision=0.9.0+cu111\
Kornia

### Dataset
For all datasets, they should be organized in below's fashion:
```
|__dataset_name
   |__Images: xxx.jpg ... 
   |__Masks : xxx.png ... 
```
For training, put your dataset folder under:
```
dataset/
```
For evaluation, download below datasets and place them under:
```
dataset/benchmark/
```
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUTS-TE](http://saliencydetection.net/duts/)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [PASCAL-S](http://cbi.gatech.edu/salobj/)
### Train & Test
**Firstly, make sure you have enough GPU RAM**.\
With default setting (batchsize=4), 24GB RAM is required, but you can always reduce the batchsize to fit your hardware.

Default values in option.py are already set to the same configuration as our paper, so\
to train your model, simply:
```
python main.py --GPU_ID 0
```
to test your model, simply:
```
python main.py --test_only --pretrain "bal_bla.pt" --GPU_ID 0
```
If you want to train/test with different settings, please refer to **option.py** for more control options.\
Currently only support training on single GPU.
### Pretrain Model & Pre-calculated Saliency Map
Our pretrain model: [[Google]](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)\
Our pre-calculated saliency map:[[Google]](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)

### Evaluation
Firstly, obtain predictions via
```
python main.py --test_only --pretrain "bal_bla.pt" --GPU_ID 0 --save_result
```
For *PR curve* and *F curve*, we use the code provided by this repo: [[BASNet, CVPR-2019]](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)\
For *MAE*, *F-score*, *E score* and *S score*, we use the code provided by this repo: [[F3Net, AAAI-2020]](https://github.com/weijun88/F3Net#evaluation)
### Citation
If you find this work is helpful, please cite our paper

### Licence
MIT License is applied.
